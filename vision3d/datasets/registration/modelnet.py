import os
import os.path as osp

import torch
import torch.utils.data
import numpy as np
import h5py
from IPython import embed
import glob
import h5py
from typing import List
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import random
import os.path as osp
import pickle
import random
from functools import partial
from sympy import im


from ...utils.point_cloud_utils import (
    random_sample_rotation, get_transform_from_rotation_translation, apply_transform
)
from ...utils.registration_utils import get_corr_indices
from ...modules.kpconv.helpers import generate_input_data, calibrate_neighbors
import open3d as o3d
from vision3d.utils.registration_utils import get_node_corr_indices_and_overlaps,to_o3d_pcd,to_array,to_tensor,compute_relative_rotation_error,pairwise_distance
from torch_scatter import scatter_min, scatter_mean, scatter_max


def load_data(partition, root):
    data_dir = os.path.join(root, '')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def uniform2sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)
class RandomCrop(object):
    """Randomly crops the *tgt_xyz* point cloud, approximately retaining half the points
    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """

    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform2sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, src, tgt, seed=None):
        if np.all(self.p_keep == 1.0):
            return src, tgt  # No need crop

        if seed is not None:
            np.random.seed(seed)

        if len(self.p_keep) == 1:
            src = self.crop(src, self.p_keep[0])
        else:
            src = self.crop(src, self.p_keep[0])
            tgt = self.crop(tgt, self.p_keep[1])
        return src, tgt

def farthest_point_sample(point, npoint, is_idx=False):
    """
    Input:
        src_xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    if is_idx:
        return point, centroids.astype(np.int32)
    return point

def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)

def translate_pc(pts):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    trans_pts = np.add(np.multiply(pts, xyz1), xyz2).astype('float32')
    return trans_pts


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd

def knn_idx(pts, k):
    kdt = cKDTree(pts)
    _, idx = kdt.query(pts, k=k + 1)
    return idx[:, 1:]

def get_rri(pts, k):
    # pts: N x 3, original points
    # q: N x K x 3, nearest neighbors
    q = pts[knn_idx(pts, k)]
    p = np.repeat(pts[:, None], k, axis=1)
    # rp, rq: N x K x 1, norms
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / rp
    qn = q / rq
    dot = np.sum(pn * qn, -1, keepdims=True)
    # theta: N x K x 1, angles
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2 * np.pi)
    idx = np.argpartition(psi, 1)[:, :, 1:2]
    # phi: N x K x 1, projection angles
    phi = np.take_along_axis(psi, idx, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1)
    return feat.reshape(-1, k * 4)


def rotation_matrix(num_axis, augment_rotation):
    """
    Sample rotation matrix along [num_axis] axis and [0 - augment_rotation] angle
    Input
        - num_axis:          rotate along how many axis
        - augment_rotation:  rotate by how many angle
    Output
        - R: [3, 3] rotation matrix
    """
    assert num_axis == 1 or num_axis == 3 or num_axis == 0
    if num_axis == 0:
        return np.eye(3)
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if num_axis == 1:
        return random.choice([Rx, Ry, Rz])
    return Rx @ Ry @ Rz


def translation_matrix(augment_translation):
    """
    Sample translation matrix along 3 axis and [augment_translation] meter
    Input
        - augment_translation:  translate by how many meters
    Output
        - t: [3, 1] translation matrix
    """
    T = np.random.rand(3) * augment_translation
    return T.reshape(3, 1)

def normalize_points(points):
    r"""Normalize point cloud to a unit sphere at origin."""
    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points
def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

class MutiModelNet(torch.utils.data.Dataset):
    def __init__(self, root, num_points=768, partition='train',
                 gaussian_noise=False, unseen=True, rot_factor=4, category=None):
        super(MutiModelNet, self).__init__()
        """ if not os.path.exists(os.path.join(root)):
            self._download_dataset(root) """
        self.data, self.label = load_data(partition, root)
        if category is not None:
            self.data = self.data[self.label == category]
            self.label = self.label[self.label == category]
        self.partition = partition
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        self.crop = RandomCrop(p_keep=[0.55, 0.55])

        if self.unseen and self.partition == 'test':
            self.data = self.data[self.label >= 20]
            self.label = self.label[self.label >= 20]
            print("unseen")
        else:
            self.data = self.data[self.label < 20]
            self.label = self.label[self.label < 20]
        self.n_points = num_points
        self.max_angle = np.pi / rot_factor
        self.max_trans = 0.5
        self.noisy = gaussian_noise
        self.k = 20
        self.get_rri = get_rri
        self.num_instance = 16
        self.max_instance_drop=12
        self.augment_axis = 3
        self.augment_rotation = 1
        self.augment_translation = 5
        aug_T=torch.zeros((3,9,3))
        aug_T[0,:,2]=-1
        aug_T[1,:,2]=0
        aug_T[2,:,2]=1
        x=torch.linspace(-1,1,3)
        x=x.reshape(-1,1)
        x=x.expand(x.shape[0],3).reshape(x.shape[0],3,1)
        y=torch.linspace(-1,1,3) 
        y=y.reshape(1,-1)
        y=y.expand(3,y.shape[1]).reshape(3,y.shape[1],1)
        xy=torch.cat((x,y),dim=-1)
        xy=xy.reshape(-1,xy.shape[-1])
        aug_T[:,:,:2]=xy
        self.aug_T=aug_T.reshape(-1,3)
    @staticmethod
    def _download_dataset(root: str):
        os.makedirs(root, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(root)))
        os.system('rm {}'.format(zipfile))
    def produce_augment(self):
        aug_R = []
        for i in range(self.num_instance):
            aug_R.append(rotation_matrix(self.augment_axis, self.augment_rotation))
        return aug_R
    def integrate_trans(self, R, t):
        """
        Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [3, 3] or [bs, 3, 3], rotation matrix
            - t: [3, 1] or [bs, 3, 1], translation matrix
        Output
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        """
        trans = np.eye(4).reshape(-1,4,4).repeat(self.num_instance,0)
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t
        return trans

    def transform(self, pts, trans):
        """
        Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
        Input
            - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        Output
            - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
        """
        trans_pts = trans[:, :3, :3] @ pts.transpose(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.transpose(0, 2, 1)
    def __getitem__(self, index):
        if self.partition != 'train':
            np.random.seed(index)
        points = self.data[index]
        src = np.random.permutation(points[:, :3])[:self.n_points]
        src_R=pairwise_distance(src,src).max()
        src/= np.sqrt(src_R) 
        if self.partition == 'train':
            pose1 = random_pose(np.pi, self.max_trans)
            src = src @ pose1[:3, :3].T 
        """ if self.subsampled:
            src, tgt = self.crop(src, tgt)
            if self.num_subsampled_points < src.shape[0]:
                src = farthest_point_sample(src, self.num_subsampled_points, is_idx=False)
                tgt = farthest_point_sample(tgt, self.num_subsampled_points, is_idx=False) """
        rand_r=torch.from_numpy(rotation_matrix(self.augment_axis, self.augment_rotation))
        trans = torch.eye(4)
        trans[:3, :3] = rand_r
        aug_T = apply_transform(self.aug_T, trans).unsqueeze(-1).numpy()
        aug_T=aug_T+np.clip(0.05 * np.random.randn(aug_T.shape[0],3,1), -1 * 0.05, 0.05)
        inds = np.random.choice(range(27), self.num_instance, replace=False) 
        aug_T=aug_T[inds]
        
        src_keypts = src.reshape(1,src.shape[0],3).repeat(self.num_instance,0)
        tgt_keypts = src_keypts + np.clip(0.01 * np.random.randn(self.num_instance,src.shape[0],3), -1 * 0.05, 0.05)
        #pointcloud num_instance, num_per_pcd, 3 :    10,256,3
        
        aug_R = self.produce_augment()
        aug_trans = self.integrate_trans(aug_R, aug_T)

        tgt_keypts = self.transform(tgt_keypts, aug_trans)
        num_instance_drop = int(self.max_instance_drop * np.random.rand() // 1)
        num_instance_input = int(self.num_instance - num_instance_drop)
        #tgt_keypts[num_instance_input:] = 1.2*self.augment_translation*np.random.rand(num_instance_drop,self.num_per_pcd,3)
        inds = np.random.choice(range(self.num_instance), num_instance_input, replace=False) 
        tgt_keypts=tgt_keypts[inds].reshape(-1,3)
        np.random.shuffle(tgt_keypts)
        np.random.shuffle(src)
        aug_trans=aug_trans[inds].astype(np.float32)
        feats0 = np.ones((tgt_keypts.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((src.shape[0], 1), dtype=np.float32)
        return tgt_keypts.astype('float32'), src.astype('float32'), feats0, feats1, aug_trans.astype('float32')

    def __len__(self):
        return self.data.shape[0]

def threedmatch_kpconv_collate_fn(list_data, config, neighborhood_limits, compute_indices=True):
    data_dicts = []

    for points0, points1, feats0, feats1, transform in list_data:
        data_dict = {}

        data_dict['transform'] = torch.tensor(transform, dtype=torch.float32)
        #data_dict['correspondences'] = torch.from_numpy(correspondences)
        data_dict['features'] = torch.from_numpy(np.concatenate([feats0, feats1], axis=0))
        data_dict['ref_points_ori'] = torch.from_numpy(points0)
        data_dict['src_points_ori'] = torch.from_numpy(points1)

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
