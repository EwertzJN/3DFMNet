import glob
import json
import os.path as osp
import random

import numpy as np
import torch
import open3d as o3d
from PIL import Image

from vision3d.datasets.registration.threedmatch_kpconv_v1 import to_o3d_pcd
from vision3d.utils.point_cloud_utils import random_sample_rotation
from vision3d.utils.registration_utils import get_corr_indices


class XYZIBDSENCEDataset(torch.utils.data.Dataset):
    """
    Loads and processes the XYZ-IBD dataset (BOP format) with an interface
    identical to the ROBISENCEDataset class.
    """

    def __init__(self,
                 vox,
                 dataset_root,
                 split,
                 matching_radius,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False,
                 camera='photoneo'):  # Added camera as it's relevant for BOP
        super(XYZIBDSENCEDataset, self).__init__()

        print(f"Initializing XYZ-IBD Dataset for split: {split}")

        # Store all parameters from the interface
        self.voxel_size = vox
        self.data_path = dataset_root
        self.partition = 'train_pbr' if split == 'train' else split
        self.matching_radius = matching_radius
        self.max_point = max_point
        self.use_augmentation = use_augmentation and (self.partition == 'train_pbr')
        self.augmentation_noise = augmentation_noise
        self.rotation_factor = rotation_factor
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated
        self.camera = camera

        # Paths and data structures
        self.obj_path = osp.join(self.data_path, 'models')
        self.frame_indexs = []
        self.sym = {}

        # 1. Load object models and symmetry information from models_info.json
        model_info_path = osp.join(self.obj_path, 'models_info.json')
        if not osp.exists(model_info_path):
            raise FileNotFoundError(f"models_info.json not found in {self.obj_path}")
        with open(model_info_path, 'r') as f:
            models_info = json.load(f)

        self.obj_ids = sorted([int(k) for k in models_info.keys()])
        for obj_id in self.obj_ids:
            obj_id_str = str(obj_id)
            # Check for either continuous or discrete symmetry keys
            if 'symmetries_continuous' in models_info[obj_id_str] or 'symmetries_discrete' in models_info[obj_id_str]:
                self.sym[obj_id] = "__SYM"
            else:
                self.sym[obj_id] = "__SYM_NONE"

        # 2. Build frame indices from BOP structure
        split_path = osp.join(self.data_path, self.partition)
        if not osp.exists(split_path):
            raise FileNotFoundError(f"Split path not found at {split_path}")

        scene_paths_raw = sorted(glob.glob(osp.join(split_path, '*')))

        # Filter for directories and ensure the directory name is a number
        scene_paths = [s for s in scene_paths_raw if osp.isdir(s) and osp.basename(s).isdigit()]

        for scene_path in scene_paths:
            scene_id = int(osp.basename(scene_path))
            gt_path = osp.join(scene_path, 'scene_gt.json') if self.partition == 'train_pbr'\
                else osp.join(scene_path, f'scene_gt_{self.camera}.json')
            if not osp.exists(gt_path):
                continue
            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)

            for view_id_str, gt_annos in scene_gt.items():
                view_id = int(view_id_str)
                # Get all unique object models present in this specific view
                objs_in_view = set(anno['obj_id'] for anno in gt_annos)
                for obj_id in objs_in_view:
                    # Create a frame index for each (model, scene_view) pair
                    self.frame_indexs.append([obj_id, scene_id, view_id])

        if len(self.frame_indexs) == 0:
            raise ValueError(f"No data frames found for split '{self.partition}'. Check dataset structure.")

        if self.partition == 'train_pbr':
            random.shuffle(self.frame_indexs)
        self.num = len(self.frame_indexs)
        self.start = 0
        print(f"Found {self.num} (model, scene_view) pairs.")

    def __len__(self):
        return self.num

    def _augment_point_cloud(self, points0, points1, Rts):
        aug_rotation = random_sample_rotation(self.rotation_factor)
        if random.random() > 0.5:
            points0 = np.matmul(points0, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(aug_rotation, rotation)
                translation = np.matmul(aug_rotation, translation)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        else:
            points1 = np.matmul(points1, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(rotation, aug_rotation.T)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        return points0, points1, Rts

    def __getitem__(self, frame_id):
        # 1. Get metadata for the current item
        id = self.start + frame_id
        frame_index = self.frame_indexs[id]
        obj_id, scene_id, view_id = frame_index

        scene_path = osp.join(self.data_path, self.partition, f"{scene_id:06d}")

        # 2. Load Scene Point Cloud (points0)
        cam_info_path = osp.join(scene_path, 'scene_camera.json') if self.partition == 'train_pbr' \
            else osp.join(scene_path, f'scene_camera_{self.camera}.json')
        depth_path = osp.join(scene_path, 'depth', f'{view_id:06d}.png') if self.partition == 'train_pbr' \
            else osp.join(scene_path, f'depth_{self.camera}', f'{view_id:06d}.png')

        with open(cam_info_path, 'r') as f:
            cam_info = json.load(f)[str(view_id)]

        cam_K = np.array(cam_info['cam_K']).reshape(3, 3)
        # from BOP docs (https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md): Multiply the depth image with this factor to get depth in mm
        depth_scale = cam_info.get('depth_scale')

        depth_raw = o3d.io.read_image(str(depth_path))
        inter = o3d.camera.PinholeCameraIntrinsic()
        rgb = np.array(Image.open(depth_path))
        inter.set_intrinsics(rgb.shape[0], rgb.shape[1], cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        pcd0 = o3d.geometry.PointCloud.create_from_depth_image( depth_raw, inter,depth_scale=1/depth_scale)
        points=np.array(pcd0.points).astype(np.float32)/1000
        point0_ori=points
        pcd0=to_o3d_pcd(points)
        pcd0=pcd0.voxel_down_sample(voxel_size=self.voxel_size)
        points0 =  np.array(pcd0.points).astype(np.float32)

        # 3. Load Model Point Cloud (points1)
        model_path = osp.join(self.obj_path, f'obj_{obj_id:06d}.ply')
        pcd1_mesh = o3d.io.read_triangle_mesh(model_path)
        points = np.array(pcd1_mesh.vertices).astype(np.float32)/1000   # Models are in mm, scale to meters
        point1_ori = points
        pcd1 = to_o3d_pcd(points)
        pcd1=pcd1.voxel_down_sample(voxel_size=self.voxel_size)
        points1 =  np.array(pcd1.points).astype(np.float32)

        # 4. Load Ground Truth Poses (Rts) and calculate correspondence overlaps
        gt_path = osp.join(scene_path, 'scene_gt.json') if self.partition == 'train_pbr' \
            else osp.join(scene_path, f'scene_gt_{self.camera}.json')
        with open(gt_path, 'r') as f:
            all_gt_annos = json.load(f)[str(view_id)]

        Rts = []
        correspondences = []
        model_point_count = float(points1.shape[0])

        obj_gt_annos = [anno for anno in all_gt_annos if anno['obj_id'] == obj_id]

        for gt_anno in obj_gt_annos:
            R_m2c = np.array(gt_anno['cam_R_m2c']).reshape(3, 3)
            t_m2c = np.array(gt_anno['cam_t_m2c']) / 1000.0  # convert mm to meters

            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = R_m2c
            transform[:3, 3] = t_m2c
            Rts.append(transform)

            correspondence=get_corr_indices(points0, points1, transform, self.matching_radius)
            if len(correspondence)==0:
                correspondences.append(len(correspondence)/model_point_count)
            else:
                correspondence=torch.from_numpy(correspondence[:,1])
                correspondence=torch.unique(correspondence)
                correspondences.append(len(correspondence)/model_point_count)

        # 5. Augment data if in training mode
        if self.use_augmentation:
            points0, points1, Rts = self._augment_point_cloud(points0, points1, Rts)

        # 6. Create dummy features
        feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)

        # 7. Return the 11-item tuple, matching ROBISENCEDataset
        return (
            points0,
            points1,
            feats0,
            feats1,
            Rts,
            torch.from_numpy(np.array(correspondences)),
            scene_id,
            self.sym[obj_id],
            frame_index,
            point0_ori,
            points1
        )


class Process_XYZIBDSENCEDataset(torch.utils.data.Dataset):
    pass
