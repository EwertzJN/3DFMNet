import os.path as osp
import pickle
import random

import torch.utils.data
import numpy as np

from ...utils.point_cloud_utils import (random_sample_rotation,
                                        get_transform_from_rotation_translation,
                                        get_rotation_translation_from_transform)
from ...utils.registration_utils import get_corr_indices

_odometry_kitti_data_split = {
    'train': ['00', '01', '02', '03', '04', '05'],
    'val': ['06', '07'],
    'test': ['08', '09', '10']
}


class OdometryKittiPairDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root,
            subset,
            matching_radius,
            max_point=30000,
            use_augmentation=True,
            augmentation_noise=0.005,
            augmentation_min_scale=0.8,
            augmentation_max_scale=1.2,
            augmentation_shift_range=2.0,
            augmentation_rotation_factor=1.,
            return_corr_indices=False
    ):
        super(OdometryKittiPairDataset, self).__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.matching_radius = matching_radius
        self.return_corr_indices = return_corr_indices
        self.max_point = max_point

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_min_scale = augmentation_min_scale
        self.aug_max_scale = augmentation_max_scale
        self.aug_shift_range = augmentation_shift_range
        self.aug_rotation_factor = augmentation_rotation_factor

        with open(osp.join(self.dataset_root, 'metadata', '{}.pkl'.format(subset)), 'rb') as f:
            self.metadata = pickle.load(f)

    def _augment_point_cloud(self, points0, points1, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        points0 = points0 + (np.random.rand(points0.shape[0], 3) - 0.5) * self.aug_noise
        points1 = points1 + (np.random.rand(points1.shape[0], 3) - 0.5) * self.aug_noise
        # random rotation
        _rotation = random_sample_rotation(self.aug_rotation_factor)
        if random.random() > 0.5:
            points0 = np.matmul(points0, _rotation.T)
            rotation = np.matmul(_rotation, rotation)
            translation = np.matmul(_rotation, translation)
        else:
            points1 = np.matmul(points1, _rotation.T)
            rotation = np.matmul(rotation, _rotation.T)
        # random scaling
        scale = self.aug_min_scale + (self.aug_max_scale - self.aug_min_scale) * random.random()
        points0 = points0 * scale
        points1 = points1 * scale
        translation = translation * scale
        # random shift
        shift0 = np.random.uniform(-self.aug_shift_range, self.aug_shift_range, 3)
        shift1 = np.random.uniform(-self.aug_shift_range, self.aug_shift_range, 3)
        points0 = points0 + shift0
        points1 = points1 + shift1
        translation = -np.matmul(shift1[np.newaxis, :], rotation.T) + translation + shift0
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return points0, points1, transform

    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if self.max_point is not None and points.shape[0] > self.max_point:
            indices = np.random.permutation(points.shape[0])[:self.max_point]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = {}
        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['frame0'] = metadata['frame0']
        data_dict['frame1'] = metadata['frame1']

        points0 = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))
        points1 = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))
        transform = metadata['transform']

        if self.use_augmentation:
            points0, points1, transform = self._augment_point_cloud(points0, points1, transform)

        if self.return_corr_indices:
            corr_indices = get_corr_indices(points0, points1, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['points0'] = points0.astype(np.float32)
        data_dict['points1'] = points1.astype(np.float32)
        data_dict['feats0'] = np.ones((points0.shape[0], 1), dtype=np.float32)
        data_dict['feats1'] = np.ones((points1.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)
