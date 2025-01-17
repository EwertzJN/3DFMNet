import os.path as osp
import os,copy
import argparse

import pickle
import random
from functools import partial
import glob

import time
from cv2 import exp

import torch
import torch.utils.data
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from torch.utils.data import Dataset
from vision3d.modules.kpconv.helpers import generate_input_data, calibrate_neighbors
from configs.config_scan2cad_focus import config
import json
from pathlib import Path
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from IPython import embed
from vision3d.utils.torch_utils import index_select
from vision3d.utils.metrics import StatisticsDictMeter, Timer
from vision3d.engine import Engine
from vision3d.utils.torch_utils import to_cuda, all_reduce_dict
from vision3d.utils.python_utils import ensure_dir

from utils.dataset import Process_Scan2cad_test_data_loader
from models.focusnet import create_model
from models.loss import FocusLoss, Evaluator
from vision3d.utils.registration_utils import get_corr_indices
from utils.transform import Transform
import pandas as pd
from vision3d.utils.point_cloud_utils import pairwise_distance, apply_transform,axis_angle_to_rotation,rotation_to_axis_angle,get_point2trans_index
from torch_cluster import knn_graph

import pinocchio as pin
import quaternion


eps = 1e-8

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, required=True, default=21)
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'extra', 'val'], default='3DMatch')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--if_proposal', type=str2bool, default=True)
    parser.add_argument('--output_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp0')
    parser.add_argument('--snapshot_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp0/snapshots')
    parser.add_argument('--logs_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp0/logs')
    parser.add_argument('--proposals_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp0/detcenter')
    parser.add_argument('--scan2cad_root', type=str, default='/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/scan2cad/scan2cad_pre/')
    parser.add_argument('--voxel_size', type=float, default=0.02)
    return parser


def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

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

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def run_one_epoch(
        engine,
        epoch,
        data_loader,
        model,
        evaluator,
        loss_func=None,
        optimizer=None,
        scheduler=None,
        training=True,
        vis_debug=False
):
    if training:
        model.train()
        if engine.distributed:
            data_loader.sampler.set_epoch(epoch)
    else:
        model.eval()

    if training:
        loss_meter = StatisticsDictMeter()
        loss_meter.register_meter('loss')
        loss_meter.register_meter('c_loss')
        loss_meter.register_meter('f_loss')
    result_meter = StatisticsDictMeter()

    timer = Timer()

    num_iter_per_epoch = len(data_loader)
    mean_time = 0
    count=0
    fail=0
    mean_cluster_instance_recall = 0.0
    mean_instance_precision = 0.0
    mean_error = 0.0


    print(num_iter_per_epoch)
    start=0
    save_number = 4705
    for i, data_dict in enumerate(data_loader):
        engine.logger.info(f'{i}')
        data_dict = to_cuda(data_dict)
        timer.add_prepare_time()
        with torch.no_grad():
            a = time.time()
            output_dict = model(data_dict)
            mean_time += time.time() - a
            pred_cluster_recall = output_dict['pred_cluster_instance_recall']
            pred_offset_error = output_dict['pred_offset_error']
            pred_cluster_precision = output_dict['pred_cluster_instance_precision']
            mean_instance_precision += pred_cluster_precision
            mean_error += pred_offset_error
            mean_cluster_instance_recall += pred_cluster_recall
            engine.logger.info(f'pred_cluster_instance_recall:{pred_cluster_recall}')
            engine.logger.info(f'pred_cluster_instance_precision:{pred_cluster_precision}')
            engine.logger.info(f'pred_offset_error:{pred_offset_error}')
            
            if engine.args.if_proposal:
                ref_fine_points = output_dict['ref_fine_points']
                src_fine_points = output_dict['src_fine_points']
                transforms = data_dict['transform']
                sym = output_dict['sym']
                for j in range(ref_fine_points.shape[0]):
                    save_ref_fine_points = ref_fine_points[j,:,:].cpu().numpy()
                    save_src_fine_points = src_fine_points[0,:,:].cpu().numpy()
                    save_transforms = transforms.cpu().numpy()
                    savename = os.path.join(engine.args.proposals_dir, 'data{:05d}.npz'.format(save_number))
                    # savename = '/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/scan2cad/detection_random_alone_4096/'+'data{:05d}.npz'.format(save_number)
                    np.savez(savename, points0=save_ref_fine_points, points1=save_src_fine_points, trans=save_transforms, frag_id1=sym, scene_name=sym, sym=sym)
                    save_number += 1
                
            count += 1
            start += 1
        
    print('pred_cluster_instance_recall', 100*mean_cluster_instance_recall/count)
    print('pred_cluster_instance_precision', 100*mean_instance_precision/count)
    print('pred_offset_error', mean_error/count)
           
            
def main():
    parser = make_parser()
    args = parser.parse_args()
    config.snapshot_dir = args.snapshot_dir
    config.logs_dir = args.logs_dir
    config.proposals_dir = args.proposals_dir
    config.process_scan2cad_root = args.scan2cad_root
    ensure_dir(config.logs_dir)
    ensure_dir(config.proposals_dir)
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        engine.args.test_epoch=46
        engine.args.benchmark='3DMatch'
        message = 'Epoch: {}, benchmark: {}'.format(engine.args.test_epoch, engine.args.benchmark)
        engine.logger.critical(message)
        message = 'Coarse config, ' + \
                  'num_proposal: {}, '.format(config.coarse_matching_num_proposal) + \
                  'dual_softmax: {}, '.format(config.coarse_matching_dual_softmax)
        engine.logger.critical(message)
        message = 'Fine config, ' + \
                  'max_num_corr: {}, '.format(config.fine_matching_max_num_corr) + \
                  'k: {}, '.format(config.fine_matching_topk) + \
                  'mutual: {}, '.format(config.fine_matching_mutual) + \
                  'with_slack: {}, '.format(config.fine_matching_with_slack) + \
                  'threshold: {:.2f}, '.format(config.fine_matching_confidence_threshold) + \
                  'conditional: {}, '.format(config.fine_matching_conditional_score) + \
                  'min_num_corr: {}'.format(config.fine_matching_min_num_corr)
        engine.logger.critical(message)

        start_time = time.time()

        data_loader =Process_Scan2cad_test_data_loader(engine, config)
        """ data_loader, _ = get_dataloader(train_dataset, config, config.test_batch_size, config.test_num_worker,
                                         shuffle=False,
                                         drop_last=False)#neighborhood_limits=neighborhood_limits, """
        #test_loader,neighborhood_limits = ROBI_test_data_loader(engine, config)
        loading_time = time.time() - start_time

        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model)

        snapshot = "./pretrain/scan2cadfocus.pth.tar"
        engine.load_snapshot(snapshot)

        start_time = time.time()
        run_one_epoch(
                engine, engine.args.test_epoch, data_loader, model, evaluator, training=False
            )
        

        loading_time = time.time() - start_time
        message = ' test_one_epoch: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)


if __name__ == '__main__':
    main()
