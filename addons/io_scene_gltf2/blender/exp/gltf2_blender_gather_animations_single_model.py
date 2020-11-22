# Copyright 2020 The glTF-Blender-IO authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file is for the Single Model/Multiple Actions animation mode.
Assumes there's one armature and mutliple actions that animate its pose bones.
Exports one glTF animation per action. Exporting is done by baking.
"""

import array
import math
import typing

import bpy
from mathutils import Matrix, Vector, Quaternion

from io_scene_gltf2.io.com import gltf2_io
from io_scene_gltf2.io.com import gltf2_io_constants
from io_scene_gltf2.io.exp import gltf2_io_binary_data
from io_scene_gltf2.blender.exp import gltf2_blender_export_keys
from io_scene_gltf2.io.com.gltf2_io_debug import print_console
from ..com.gltf2_blender_extras import generate_extras


def gather_animations(scene, export_settings):
    arma_node, found_multiple = __find_single_arma_node(scene)
    if found_multiple:
        print_console("WARNING",
            "Single Model/Multiple Action mode does not work with multiple armatures. "
            "No animations will be exported.")
        return []
    if arma_node is None:
        return []
    arma_ob = arma_node.__blender_data[1]

    actions = __get_actions()
    if not actions:
        return []

    bone_nodes = __get_bone_nodes_for_arma_node(arma_node)

    # Basic structure here will be...
    #    for each action:
    #        put action on a temp NLA track and star it:
    #        for each frame in frame range:
    #            seek to frame
    #            read all the pose bone positions
    #        make animation from that data
    #    restore original animation state

    animations = []

    needs_animation_data_clear = False
    orig_solo_track = None
    orig_frame = None
    orig_subframe = None
    tmp_nla_track = None
    try:
        if arma_ob.animation_data is None:
            arma_ob.animation_data_create()
            needs_animation_data_clear = True

        tmp_nla_track = arma_ob.animation_data.nla_tracks.new()

        for track in arma_ob.animation_data.nla_tracks:
            if track.is_solo:
                orig_solo_track = track
                break

        orig_frame = bpy.context.scene.frame_current
        orig_subframe = bpy.context.scene.frame_subframe

        for action in actions:
            anim = __gather_animation(arma_ob, bone_nodes, tmp_nla_track, action, export_settings)
            animations.append(anim)

    finally:
        # Restore original animation state
        if needs_animation_data_clear:
            arma_ob.animation_data_clear()
        else:
            if orig_solo_track is None:
                arma_ob.animation_data.nla_tracks[0].is_solo = True
                arma_ob.animation_data.nla_tracks[0].is_solo = False
            else:
                orig_solo_track.is_solo = True

            if tmp_nla_track is not None:
                arma_ob.animation_data.nla_tracks.remove(tmp_nla_track)

            if orig_subframe is not None:
                bpy.context.scene.frame_set(orig_frame, subframe=orig_subframe)

    return animations


def __gather_animation(arma_ob, bone_nodes, nla_track, action, export_settings):
    # Clear old action from the track
    while nla_track.strips:
        nla_track.strips.remove(nla_track.strips[0])

    # Clear all pose bone transforms; this prevents the current animation
    # being affected by old values from the last one
    for pbone in arma_ob.pose.bones:
        pbone.location = (0, 0, 0)
        pbone.rotation_euler = (0, 0, 0)
        pbone.rotation_quaternion = (1, 0, 0, 0)
        pbone.scale = (1, 1, 1)

    # Put the action on the track and star it so it plays
    strip = nla_track.strips.new('tmp', start=0, action=action)
    nla_track.is_solo = True

    # Seek to each frame and read the bone TRS data

    frame_start = math.floor(strip.frame_start)
    frame_end = math.ceil(strip.frame_end)
    frame_step = export_settings['gltf_frame_step']

    bone_names = [node.__blender_data[2] for node in bone_nodes]
    bone_data = {}
    bone_data["translation"] = {bone_name: [] for bone_name in bone_names}
    bone_data["rotation"] = {bone_name: [] for bone_name in bone_names}
    bone_data["scale"] = {bone_name: [] for bone_name in bone_names}

    for frame in range(frame_start, frame_end, frame_step):
        bpy.context.scene.frame_set(frame)
        for bone_name in bone_names:
            pbone = arma_ob.pose.bones[bone_name]
            t, r, s = __get_gltf_trs_from_bone(pbone, arma_ob, export_settings)
            bone_data["translation"][bone_name].append(t)
            bone_data["rotation"][bone_name].append(r)
            bone_data["scale"][bone_name].append(s)

    # Put it all together to get the glTF animation

    channels = []
    samplers = []

    input_accessor = __get_keyframe_accessor(frame_start, frame_end, frame_step)

    for bone_node in bone_nodes:
        _, _, bone_name = bone_node.__blender_data
        for path in ['translation', 'rotation', 'scale']:
            sampler = gltf2_io.AnimationSampler(
                input=input_accessor,
                interpolation=None,  # LINEAR
                output=__encode_output_accessor(bone_data[path][bone_name]),
                extensions=None,
                extras=None,
            )
            samplers.append(sampler)
            channel = gltf2_io.AnimationChannel(
                sampler=len(samplers) - 1,
                target=gltf2_io.AnimationChannelTarget(
                    node=bone_node,
                    path=path,
                    extensions=None,
                    extras=None,
                ),
                extensions=None,
                extras=None,
            )
            channels.append(channel)

    animation = gltf2_io.Animation(
        channels=channels,
        extensions=None,
        extras=__gather_extras(action, export_settings),
        name=action.name,
        samplers=samplers,
    )

    return animation


def __get_gltf_trs_from_bone(pbone, arma_ob, export_settings):
    # This is just copied from gltf2_blender_gather_joints, let's not
    # pretend I know how it works.

    # XXX: suspect this is buggy; gltf2_blender_gather_nodes also mucks
    # with bone transforms doesn't it?

    axis_basis_change = Matrix.Identity(4)
    if export_settings[gltf2_blender_export_keys.YUP]:
        axis_basis_change = Matrix(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))

    if pbone.parent is None:
        correction_matrix_local = axis_basis_change @ pbone.bone.matrix_local
    else:
        correction_matrix_local = (
            pbone.parent.bone.matrix_local.inverted() @
            pbone.bone.matrix_local
        )

    if (pbone.bone.use_inherit_rotation == False or pbone.bone.inherit_scale != "FULL") and pbone.parent != None:
        rest_mat = (pbone.parent.bone.matrix_local.inverted_safe() @ pbone.bone.matrix_local)
        matrix_basis = (rest_mat.inverted_safe() @ pbone.parent.matrix.inverted_safe() @ pbone.matrix)
    else:
        matrix_basis = pbone.matrix
        matrix_basis = arma_ob.convert_space(pose_bone=pbone, matrix=matrix_basis, from_space='POSE', to_space='LOCAL')

    t, r, s = (correction_matrix_local @ matrix_basis).decompose()
    r = Quaternion((r[1], r[2], r[3], r[0]))  # wxyz -> xyzw

    return t, r, s


def __find_single_arma_node(scene):
    # Recursively scan a scene looking for a single armature.
    arma_node = None
    found_multiple = False

    def visit(node):
        nonlocal arma_node, found_multiple
        if hasattr(node, '__blender_data'):
            if node.__blender_data[0] == 'OBJECT':
                ob = node.__blender_data[1]
                if ob.type == 'ARMATURE':
                    if arma_node is None:
                        arma_node = node
                    else:
                        found_multiple = True

        for child in node.children:
            visit(child)

    for node in scene.nodes:
        visit(node)

    return arma_node, found_multiple


def __get_actions():
    # Get all the actions we want to export.
    def had_pose_bone_fcurve(action):
        return any(fc.data_path.startswith('pose.bone') for fc in action.fcurves)

    return [
        action for action in bpy.data.actions
        if had_pose_bone_fcurve(action)
    ]


def __get_bone_nodes_for_arma_node(arma_node):
    bones = []
    def visit(node):
        if hasattr(node, '__blender_data'):
            if node.__blender_data[0] == 'BONE':
                bones.append(node)
                for child in node.children:
                    visit(child)

    for child in arma_node.children:
        visit(child)

    return bones


def __gather_extras(action, export_settings):
    if export_settings['gltf_extras']:
        return generate_extras(action)
    return None


def __get_keyframe_accessor(frame_start, frame_end, frame_step):
    # Gets an accessor for a range of keyframes. Used for sampler.input.
    fps = bpy.context.scene.render.fps
    keyframes = [frame / fps for frame in range(frame_start, frame_end, frame_step)]
    keyframe_data = array.array('f', keyframes).tobytes()
    return gltf2_io.Accessor(
        buffer_view=gltf2_io_binary_data.BinaryData(keyframe_data),
        component_type=gltf2_io_constants.ComponentType.Float,
        type=gltf2_io_constants.DataType.Scalar,
        count=len(keyframes),
        min=[keyframes[0]],
        max=[keyframes[-1]],
        byte_offset=None,
        extensions=None,
        extras=None,
        name=None,
        normalized=None,
        sparse=None,
    )


def __encode_output_accessor(values):
    # Encodes a list of T, R, or S (Vector, Quaternion) values to an accessor.
    vals = [x for val in values for x in val]
    vals_data = array.array('f', vals).tobytes()
    return gltf2_io.Accessor(
        buffer_view=gltf2_io_binary_data.BinaryData(vals_data),
        component_type=gltf2_io_constants.ComponentType.Float,
        type=gltf2_io_constants.DataType.vec_type_from_num(len(values[0])),
        count=len(values),
        min=None,
        max=None,
        byte_offset=None,
        extensions=None,
        extras=None,
        name=None,
        normalized=None,
        sparse=None,
    )
