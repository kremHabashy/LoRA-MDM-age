from os.path import join as pjoin
import importlib.util as _importlib_util

from data_loaders.humanml.common.skeleton import Skeleton
import numpy as np
import os
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *

import torch
from tqdm import tqdm
import argparse
import sys

from torch.utils.data import Dataset

class Text2MotionDataset(Dataset):
    def __init__(self, data_dir, split='train', num_frames=196):
        self.data_dir = data_dir
        self.split = split
        self.num_frames = num_frames

        self.motion_dir = os.path.join(data_dir, "motions")
        with open(os.path.join(data_dir, f"{split}.txt")) as f:
            self.motion_names = [line.strip() for line in f]

        self.action_list = None
        self.label_map = {}
        self.num_actions = 12

        if os.path.exists(os.path.join(data_dir, "actions.txt")):
            with open(os.path.join(data_dir, "actions.txt")) as f:
                self.action_list = [line.strip() for line in f]
            self.label_map = {name: i for i, name in enumerate(self.action_list)}

        self.mean = np.load(os.path.join(data_dir, "Mean.npy"))
        self.std = np.load(os.path.join(data_dir, "Std.npy"))

    def __len__(self):
        return len(self.motion_names)

    def __getitem__(self, index):
        motion_name = self.motion_names[index]
        motion = np.load(os.path.join(self.motion_dir, f"{motion_name}.npy"))
        motion = (motion - self.mean) / self.std
        motion = motion[:self.num_frames]
        motion = torch.from_numpy(motion).float()

        # handle label if available
        label = None
        if self.action_list is not None:
            action_name = "_".join(motion_name.split("_")[:-1])
            label = self.label_map[action_name]
            label = torch.tensor([label]).long()

        return {
            "motion": motion,       # [T, D]
            "length": motion.shape[0],
            "text": motion_name,    # Not really used here
            "action": label         # Used for conditioning
        }

class Text2MotionDatasetV2(Text2MotionDataset):
    def __init__(self, data_dir, split='train', num_frames=196, styles=None):
        super().__init__(data_dir=data_dir, split=split, num_frames=num_frames)
        self.styles = styles

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.styles:
            item["style"] = self.styles[0]  # style name used for LoRA
        return item



# positions (batch, joint_num, 3)
def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    global_positions = positions.copy()
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_rot(data):
    # dataset [bs, seqlen, 263/251] HumanML/KIT
    joints_num = 22 if data.shape[-1] == 263 else 21
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
    return cont6d_params


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def _load_vc_metadata(meta_py_path):
    """Dynamically import your metadata.py and build subject dicts."""
    spec = _importlib_util.spec_from_file_location("vcmeta", meta_py_path)
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    able = mod.create_able_bodied_metadata()
    stroke = mod.create_stroke_metadata()
    return able, stroke

def _subject_from_relpath(rel):  # e.g., "SUBJ01/SUBJ1_0_humanml3d_22joints"
    return rel.split(os.sep, 1)[0]

def _compose_caption(subj_id, md):
    # Compose a compact HumanML-style caption from metadata.
    base = "an adult walking."
    if not md:
        return base
    extras = []
    # age = md.get("age")
    # if isinstance(age, (int, float)) and age > 0:
    #     extras.append(f"age {int(age)} years")
    # h = md.get("height_m")
    # if isinstance(h, (int, float)) and h > 0:
    #     extras.append(f"height {h:.2f} m")
    # leg = md.get("leg_length_m")
    # if isinstance(leg, (int, float)) and leg > 0:
    #     extras.append(f"leg length {leg:.2f} m")
    # sp = md.get("walking_speeds", {})
    # ls, rs = sp.get("left_speed"), sp.get("right_speed")
    # if isinstance(ls, (int, float)) and isinstance(rs, (int, float)):
    #     extras.append(f"speed {0.5*(ls+rs):.2f} m/s")
    return f"{base} ({', '.join(extras)})" if extras else base

def _safe_age_from_meta(subj_id, able_md, stroke_md):
    d = stroke_md.get(subj_id, {}) if subj_id.startswith("TVC") else able_md.get(subj_id, {})
    a = d.get("age", -1)
    try:
        return float(a) if a is not None else -1.0
    except Exception:
        return -1.0


def _load_joints(npz_obj):
    # Try common keys
    for k in ('joints', 'keypoints3d', 'xyz', 'J'):
        if k in npz_obj.files:
            arr = npz_obj[k]
            break
    else:
        raise KeyError(f"No joints array found in {list(npz_obj.files)}")
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 66:  # flattened T x (22*3)
        arr = arr.reshape(-1, 22, 3)
    assert arr.ndim == 3 and arr.shape[1] in (21, 22) and arr.shape[2] == 3, \
        f"Unexpected joints shape {arr.shape} (expect [T,22,3] or [T,21,3])"
    return arr


def build_vc_dataset(vc_root: str):
    """
    Convert VC (22-joint) NPZ into HumanML3D 263-D features using split lists.
    Expected input:
      vc_root/Comp_v6_KLD01/SUBJxx/*.npz
      and split lists in args.vc_splits_dir/{train,val,test}.txt
    """
    import glob
    global tgt_offsets, n_raw_offsets, kinematic_chain, fid_r, fid_l, face_joint_indx
    global l_idx1, l_idx2

    base = os.path.join(vc_root, 'Comp_v6_KLD01')
    splits_dir = args.vc_splits_dir

    # ---- T2M (HumanML3D) skeleton config (22 joints) ----
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    # Lower legs indices in 22-joint t2m order:
    l_idx1, l_idx2 = 5, 8
    # feet id (right/left) in 22-joint t2m order:
    fid_r, fid_l = [8, 11], [7, 10]
    # face joints indices (r_hip, l_hip, r_shoulder, l_shoulder)
    face_joint_indx = [2, 1, 17, 16]

    # ---- Read split files ----
    split_names = []
    for s in ('train', 'val', 'test'):
        fpath = os.path.join(splits_dir, f'{s}.txt')
        if os.path.exists(fpath):
            with open(fpath) as f:
                rels = [ln.strip() for ln in f if ln.strip()]
            # expand to absolute paths under base
            files = []
            for rel in rels:
                p = os.path.join(base, rel)
                if not p.endswith('.npz'):
                    p += '.npz'
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Listed in {s}.txt but missing on disk: {p}")
                files.append(p)
            split_names.append((s, files))

    # Fallback: if no split files at all, dump everything into 'train'
    if not split_names:
        all_npz = sorted(glob.glob(os.path.join(base, 'SUBJ*', '*.npz')))
        split_names = [('train', all_npz)]

    # ---- Compute target offsets from the first file ----
    sample_file = split_names[0][1][0]
    ex = np.load(sample_file, allow_pickle=True)
    joints0 = _load_joints(ex)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(joints0[0]))

    # ---- Outputs ----
    meta_dir = os.path.join(base, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    train_feats = []

    able_md, stroke_md = _load_vc_metadata(args.vc_meta_py)

    for split, files in split_names:
        out_motion = os.path.join(base, split, 'motions')
        out_text   = os.path.join(base, split, 'texts')
        out_age    = os.path.join(base, split, 'ages')
        os.makedirs(out_motion, exist_ok=True)
        os.makedirs(out_text, exist_ok=True)
        os.makedirs(out_age, exist_ok=True)

        names = []
        for fin in tqdm(files, desc=f"VC→HumanML3D [{split}]"):
            d = np.load(fin, allow_pickle=True)
            joints = _load_joints(d).astype(np.float32)

            feats, _, _, _ = process_file(joints, 0.002)   # -> (T-1, 263)
            assert feats.shape[1] == 263, f"Expected 263 features, got {feats.shape[1]} for {fin}"

            # relative name WITHOUT extension, keep subject subdir to avoid name clashes
            rel = os.path.splitext(os.path.relpath(fin, base))[0]  # e.g. SUBJ01/SUBJ1_0_humanml3d_22joints
            subj_id = _subject_from_relpath(rel)

            # where to save
            dest_m = os.path.join(out_motion, rel + '.npy')
            dest_t = os.path.join(out_text,   rel + '.txt')
            dest_a = os.path.join(out_age,    rel + '.txt')
            os.makedirs(os.path.dirname(dest_m), exist_ok=True)
            os.makedirs(os.path.dirname(dest_t), exist_ok=True)
            os.makedirs(os.path.dirname(dest_a), exist_ok=True)

            np.save(dest_m, feats.astype(np.float32))

            subj_meta = (stroke_md if subj_id.startswith("TVC") else able_md).get(subj_id, {})
            caption = _compose_caption(subj_id, subj_meta)
            with open(dest_t, 'w') as f_txt:
                f_txt.write(caption)

            age_val = _safe_age_from_meta(subj_id, able_md, stroke_md)
            with open(dest_a, 'w') as f_age:
                f_age.write(str(age_val))


            # the name string that HumanML loader will use
            names.append(rel)

            if split == 'train':
                train_feats.append(feats)

        # write split list at dataset root (what HumanML loader expects)
        with open(os.path.join(base, f'{split}.txt'), 'w') as f_split:
            f_split.write('\n'.join(names))

    # ---- Stats over train ----
    if train_feats:
        cat = np.concatenate(train_feats, axis=0)
        mean = cat.mean(axis=0).astype(np.float32)
        std  = cat.std(axis=0).astype(np.float32)
        np.save(os.path.join(meta_dir, 'mean.npy'), mean)
        np.save(os.path.join(meta_dir, 'std.npy'),  std)
        # Some code paths expect capitalized copies at data_root
        np.save(os.path.join(base, 'Mean.npy'), mean)
        np.save(os.path.join(base, 'Std.npy'),  std)

def add_vc_ages_and_placeholders(vc_base: str, meta_py_path: str, overwrite: bool = True):
    """
    Populate ages/ and texts/ for an already-converted VC dataset.

    Expects layout like:
      vc_base/
        train/
          motions/SUBJxx/<clip>.npy
          [texts/... will be created/updated]
          [ages/...  will be created/updated]
        val/...
        test/...

    We DO NOT touch motion files or recompute stats/splits.
    """
    import glob

    able_md, stroke_md = _load_vc_metadata(meta_py_path)

    def rel_from_motion(motion_path: str, split: str) -> str:
        # e.g. ".../van_criekinge/train/motions/SUBJ03/SUBJ3_0_humanml3d_22joints.npy"
        # -> "SUBJ03/SUBJ3_0_humanml3d_22joints"
        stem = os.path.splitext(os.path.relpath(motion_path,
                    os.path.join(vc_base, split, 'motions')))[0]
        return stem

    splits = [s for s in ('train', 'val', 'test')
              if os.path.isdir(os.path.join(vc_base, s, 'motions'))]

    for split in splits:
        motions_dir = os.path.join(vc_base, split, 'motions')
        texts_dir   = os.path.join(vc_base, split, 'texts')
        ages_dir    = os.path.join(vc_base, split, 'ages')
        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(ages_dir,  exist_ok=True)

        motion_files = sorted(glob.glob(os.path.join(motions_dir, '**', '*.npy'), recursive=True))
        for mpath in tqdm(motion_files, desc=f"Add ages/placeholders [{split}]"):
            rel   = rel_from_motion(mpath, split)
            subj  = _subject_from_relpath(rel)
            age_f = os.path.join(ages_dir,  rel + '.txt')
            txt_f = os.path.join(texts_dir, rel + '.txt')
            os.makedirs(os.path.dirname(age_f), exist_ok=True)
            os.makedirs(os.path.dirname(txt_f), exist_ok=True)

            # AGE
            age_val = _safe_age_from_meta(subj, able_md, stroke_md)
            if overwrite or not os.path.exists(age_f):
                with open(age_f, 'w') as fa:
                    fa.write(str(age_val))

            # PLACEHOLDER TEXT
            if overwrite or not os.path.exists(txt_f):
                with open(txt_f, 'w') as ft:
                    ft.write('An adult is walking.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_vc", action="store_true",
                        help="Convert VC joints to HumanML3D features")
    parser.add_argument("--vc_root", type=str, default="dataset/HumanML3D")
    parser.add_argument("--vc_splits_dir", type=str, default="/u1/khabashy/LoRA-MDM/dataset/vc",
                    help="Dir with {train,val,test}.txt listing relative NPZ paths like 'SUBJ01/xxx.npz'")
    parser.add_argument("--add_vc_ages_text", action="store_true",
                    help="Create/overwrite ages/ and texts/ (placeholder) for existing VC features.")
    parser.add_argument("--vc_base", type=str,
                        help="Path to base of your converted VC dataset (e.g. .../data/humanml3d/van_criekinge).")
    parser.add_argument("--vc_meta_py", type=str,
                        default="/u1/khabashy/LoRA-MDM/data/van_criekinge/metadata.py",
                        help="Path to metadata.py containing subject ages.")


    args = parser.parse_args()
    if args.build_vc:
        build_vc_dataset(args.vc_root)

    if args.add_vc_ages_text:
        if not args.vc_base:
            raise ValueError("--vc_base is required with --add_vc_ages_text")
        add_vc_ages_and_placeholders(args.vc_base, args.vc_meta_py, overwrite=True)

    sys.exit(0)
