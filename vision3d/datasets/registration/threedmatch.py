import os.path as osp
import pickle
import random

import torch
import torch.utils.data
import numpy as np

from ...utils.point_cloud_utils import (random_sample_rotation,
                                        get_transform_from_rotation_translation)
from ...utils.registration_utils import get_corr_indices


class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root,
            subset,
            matching_radius,
            max_num_point=30000,
            use_augmentation=True,
            augmentation_noise=0.005,
            augmentation_rotation_factor=1,
            overlap_thresh=None,
            return_corr_indices=True,
            rotated=False
    ):
        super(ThreeDMatchPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.matching_radius = matching_radius
        self.max_num_point = max_num_point
        self.return_corr_indices = return_corr_indices
        self.overlap_thresh = overlap_thresh
        self.rotated = rotated

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation_factor = augmentation_rotation_factor

        with open(osp.join(self.metadata_root, subset + '_v3.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
            if self.overlap_thresh is not None:
                self.metadata = [metadata for metadata in self.metadata if metadata['overlap'] > self.overlap_thresh]

        if self.rotated:
            self.rotations = {}
            for metadata in self.metadata:
                frame_names = [metadata['scene_name'] + '/' + metadata['frag_id' + i] for i in ['0', '1']]
                for frame_name in frame_names:
                    if frame_name not in self.rotations:
                        self.rotations[frame_name] = random_sample_rotation(1.)

    def __len__(self):
        return len(self.metadata)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        if self.max_num_point is not None and points.shape[0] > self.max_num_point:
            indices = np.random.permutation(points.shape[0])[:self.max_num_point]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""
        ref_points = src_points @ rotation.T + translation
        """
        ref_rotation = random_sample_rotation(self.aug_rotation_factor)
        ref_points = np.matmul(ref_points, ref_rotation.T)
        rotation = np.matmul(ref_rotation, rotation)
        translation = np.matmul(ref_rotation, translation)

        src_rotation = random_sample_rotation(self.aug_rotation_factor)
        src_points = np.matmul(src_points, src_rotation.T)
        rotation = np.matmul(rotation, src_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata = self.metadata[index]
        scene_name = metadata['scene_name']
        ref_frame = metadata['frag_id0']
        src_frame = metadata['frag_id1']

        data_dict['scene_name'] = scene_name
        data_dict['ref_frame'] = ref_frame
        data_dict['src_frame'] = src_frame
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get pointcloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = self.rotations[scene_name + '/' + str(ref_frame)]
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = self.rotations[scene_name + '/' + str(src_frame)]
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_corr_indices(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
