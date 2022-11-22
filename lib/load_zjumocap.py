import os
import glob
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

def load_canonical_joints(self):
    cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
    with open(cl_joint_path, 'rb') as f:
        cl_joint_data = pickle.load(f)
    canonical_joints = cl_joint_data['joints'].astype('float32')
    canonical_bbox = self.skeleton_to_bbox(canonical_joints)

    return canonical_joints, canonical_bbox

def load_train_cameras(self):
    cameras = None
    with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
        cameras = pickle.load(f)
    return cameras

@staticmethod
def skeleton_to_bbox(skeleton):
    min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
    max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

    return {
        'min_xyz': min_xyz,
        'max_xyz': max_xyz
    }

def load_train_mesh_infos(self):
    mesh_infos = None
    with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
        mesh_infos = pickle.load(f)

    for frame_name in mesh_infos.keys():
        bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
        mesh_infos[frame_name]['bbox'] = bbox

    return mesh_infos

def load_train_frames(self):
    img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                            exts=['.png'])
    return [split_path(ipath)[1] for ipath in img_paths]

def query_dst_skeleton(self):
    return {
        'poses': self.train_mesh_info['poses'].astype('float32'),
        'dst_tpose_joints': \
            self.train_mesh_info['tpose_joints'].astype('float32'),
        'bbox': self.train_mesh_info['bbox'].copy(),
        'Rh': self.train_mesh_info['Rh'].astype('float32'),
        'Th': self.train_mesh_info['Th'].astype('float32')
    }

def get_freeview_camera(self, frame_idx, total_frames, trans=None):
    E = rotate_camera_by_frame_idx(
            extrinsics=self.train_camera['extrinsics'], 
            frame_idx=frame_idx,
            period=total_frames,
            trans=trans,
            **self.ROT_CAM_PARAMS[self.src_type])
    K = self.train_camera['intrinsics'].copy()
    K[:2] *= cfg.resize_img_scale
    return K, E

def load_image(self, frame_name, bg_color):
    imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
    orig_img = np.array(load_image(imagepath))

    maskpath = os.path.join(self.dataset_path, 
                            'masks', 
                            '{}.png'.format(frame_name))
    alpha_mask = np.array(load_image(maskpath))
    
    if 'distortions' in self.train_camera:
        K = self.train_camera['intrinsics']
        D = self.train_camera['distortions']
        orig_img = cv2.undistort(orig_img, K, D)
        alpha_mask = cv2.undistort(alpha_mask, K, D)

    alpha_mask = alpha_mask / 255.
    img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
    if cfg.resize_img_scale != 1.:
        img = cv2.resize(img, None, 
                            fx=cfg.resize_img_scale,
                            fy=cfg.resize_img_scale,
                            interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = cv2.resize(alpha_mask, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LINEAR)
                            
    return img, alpha_mask




def load_zju_data(basedir):
    
    return imgs, poses, render_poses, [H, W, focal], K, i_split