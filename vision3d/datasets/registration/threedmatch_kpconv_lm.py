import os.path as osp
import pickle
import random
from functools import partial

import torch
import torch.utils.data
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import json
from pathlib import Path
from PIL import Image



from ...utils.point_cloud_utils import (
    random_sample_rotation, get_transform_from_rotation_translation, apply_transform
)
from ...utils.registration_utils import get_corr_indices
from ...modules.kpconv.helpers import generate_input_data, calibrate_neighbors
import open3d as o3d

import pinocchio as pin
from transform import Transform

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def remap_bop_targets(targets):
    targets = targets.rename(columns={'im_id': 'view_id'})
    targets['label'] = targets['obj_id'].apply(lambda x: f'obj_{x:06d}')
    return targets


def build_index(ds_dir, save_file, subset, save_file_annotations):
    scene_ids, cam_ids, view_ids = [], [], []

    annotations = dict()
    base_dir = ds_dir / subset

    for scene_dir in base_dir.iterdir():
        scene_id = scene_dir.name
        annotations_scene = dict()
        for f in ('scene_camera.json', 'scene_gt_info.json', 'scene_gt.json'):
            path = (scene_dir / f)
            if path.exists():
                annotations_scene[f.subset('.')[0]] = json.loads(path.read_text())
        annotations[scene_id] = annotations_scene
        # for view_id in annotations_scene['scene_gt_info'].keys():
        for view_id in annotations_scene['scene_camera'].keys():
            cam_id = 'cam'
            scene_ids.append(int(scene_id))
            cam_ids.append(cam_id)
            view_ids.append(int(view_id))

    frame_index = pd.DataFrame({'scene_id': scene_ids, 'cam_id': cam_ids,
                                'view_id': view_ids, 'cam_name': cam_ids})
    frame_index.to_feather(save_file)
    save_file_annotations.write_bytes(pickle.dumps(annotations))
    return

class ThreeDMatchPairKPConvDataset(torch.utils.data.Dataset):
    def __init__(self,
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
                 rotated=False):
        super(ThreeDMatchPairKPConvDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = split
        self.matching_radius = matching_radius
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.rotation_factor = rotation_factor
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated

        ds_dir = Path(dataset_root)
        self.ds_dir = ds_dir
        assert ds_dir.exists(), 'Dataset does not exists.'

        self.split = split
        self.base_dir = ds_dir / split

        save_file_index = self.ds_dir / f'index_{split}.feather'
        save_file_annotations = self.ds_dir / f'annotations_{split}.pkl'

        build_index(ds_dir=ds_dir, save_file=save_file_index,save_file_annotations=save_file_annotations,split=split)
        
        self.frame_index = pd.read_feather(save_file_index).reset_index(drop=True)
        self.annotations = pickle.loads(save_file_annotations.read_bytes())

        models_infos = json.loads((ds_dir / 'models' / 'models_info.json').read_text())
        self.all_labels = [f'obj_{int(obj_id):06d}' for obj_id in models_infos.keys()]

        objects = []
        for obj_id, bop_info in models_infos.items():
            obj_id = int(obj_id)
            obj_label = f'obj_{obj_id:06d}'
            mesh_path = (ds_dir / 'models' / obj_label).with_suffix('.ply').as_posix()
            obj = dict(
                label=obj_label,
                category=None,
                mesh_path=mesh_path,
                mesh_units='mm',
            )
            is_symmetric = False
            for k in ('symmetries_discrete', 'symmetries_continuous'):
                obj[k] = bop_info.get(k, [])
                if len(obj[k]) > 0:
                    is_symmetric = True
            obj['is_symmetric'] = is_symmetric
            obj['diameter'] = bop_info['diameter']
            scale = 0.001 if obj['mesh_units'] == 'mm' else 1.0
            obj['diameter_m'] = bop_info['diameter'] * scale
            objects.append(obj)
        self.objects = objects

    def __len__(self):
        return len(self.frame_index)


    def __getitem__(self, frame_id):
        # metadata

        row = self.frame_index.iloc[frame_id]
        scene_id, view_id = row.scene_id, row.view_id
        view_id = int(view_id)
        view_id_str = f'{view_id:06d}'
        scene_id_str = f'{int(scene_id):06d}'
        scene_dir = self.base_dir / scene_id_str

        depth_dir =scene_dir/ 'depth'
        depth_path = depth_dir / f'{view_id_str}.png'
        rgb = np.array(Image.open(depth_path))

        depth_raw = o3d.io.read_image(str(depth_path))
        inter = o3d.camera.PinholeCameraIntrinsic()
        cam_annotation = self.annotations[scene_id_str]['scene_camera'][str(view_id)]

        if 'cam_R_w2c' in cam_annotation:
            RC0 = np.array(cam_annotation['cam_R_w2c']).reshape(3, 3)
            tC0 = np.array(cam_annotation['cam_t_w2c']) * 0.001
            TC0 = Transform(RC0, tC0)
        else:
            TC0 = Transform(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation['cam_K']).reshape(3, 3)
        T0C = TC0.inverse()
        T0C = T0C.toHomogeneousMatrix()
        inter.set_intrinsics(rgb.shape[0], rgb.shape[1], K[0][0], K[1][1], K[0][2], K[1][2])
        T0C = TC0.inverse()
        dataset = []
        #mask = np.zeros((h, w), dtype=np.uint8)
        if 'scene_gt_info' in self.annotations[scene_id_str]:
            annotation = self.annotations[scene_id_str]['scene_gt'][str(view_id)]
            n_objects = len(annotation)
            visib = self.annotations[scene_id_str]['scene_gt_info'][str(view_id)]
            for n in range(n_objects):

                RCO = np.array(annotation[n]['cam_R_m2c']).reshape(3, 3)
                tCO = np.array(annotation[n]['cam_t_m2c']) * 0.001
                TCO = Transform(RCO, tCO)
                T0O = T0C * TCO
                T0O = T0O.toHomogeneousMatrix()
                
                obj_id = annotation[n]['obj_id']-1
                pcd0 = o3d.geometry.PointCloud.create_from_depth_image( depth_raw, inter,depth_scale=1000) 
                pcd0=pcd0.voxel_down_sample(voxel_size=0.005)          
                pcd1=o3d.io.read_point_cloud(self.objects[obj_id]['mesh_path'])
                pcd1=pcd1.voxel_down_sample(voxel_size=5)       
                
                points0 =  np.array(pcd0.points).astype(np.float32)
                points1 =  np.array(pcd1.points).astype(np.float32)/1000
                feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
                feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)
                scene_name=scene_dir
                frag_id0=view_id
                frag_id1=obj_id
                mask_n = np.array(Image.open(scene_dir / 'mask_visib' / f'{view_id_str}_{n:06d}.png'))

                correspondences = get_corr_indices(points0, points1, T0O, self.matching_radius)
                obj=points0,points1,feats0,feats1,T0O,correspondences,scene_name, frag_id0, frag_id1
            dataset.append(obj)
                
        return dataset



def threedmatch_kpconv_collate_fn(list_data, config, neighborhood_limits, compute_indices=True):
    data_dicts = []

    for points0, points1, feats0, feats1, transform, correspondences, scene_name, frag_id0, frag_id1 in list_data:
        data_dict = {}

        data_dict['scene_name'] = scene_name
        data_dict['frag_id0'] = frag_id0
        data_dict['frag_id1'] = frag_id1
        data_dict['transform'] = torch.from_numpy(transform)
        #data_dict['correspondences'] = torch.from_numpy(correspondences)
        data_dict['correspondences'] =correspondences
        data_dict['features'] = torch.from_numpy(np.concatenate([feats0, feats1], axis=0))

        stacked_points = torch.from_numpy(np.concatenate([points0, points1], axis=0))
        stacked_lengths = torch.from_numpy(np.array([points0.shape[0], points1.shape[0]]))

        if compute_indices:
            input_points, input_neighbors, input_pools, input_upsamples, input_lengths = generate_input_data(
                stacked_points, stacked_lengths, config, neighborhood_limits
            )

            data_dict['points'] = input_points
            data_dict['neighbors'] = input_neighbors
            data_dict['pools'] = input_pools
            data_dict['upsamples'] = input_upsamples
            data_dict['stack_lengths'] = input_lengths
        else:
            data_dict['stacked_points'] = stacked_points
            data_dict['stacked_lengths'] = stacked_lengths

        data_dicts.append(data_dict)

    if len(data_dicts) == 1:
        return data_dicts[0]
    else:
        return data_dicts


def get_dataloader(
        dataset,
        config,
        batch_size,
        num_workers,
        shuffle=False,
        neighborhood_limits=None,
        drop_last=True,
        sampler=None,
        compute_indices=True
):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=threedmatch_kpconv_collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=partial(
            threedmatch_kpconv_collate_fn, config=config, neighborhood_limits=neighborhood_limits,
            compute_indices=compute_indices
        ),
        drop_last=drop_last
    )
    return dataloader, neighborhood_limits
