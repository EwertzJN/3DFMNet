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
from configs.config_robi_match import config
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

from utils.dataset import Process_ROBI_test_data_loader
from models.matchnet_robi import create_model
from models.loss import MatchLoss, Evaluator
from vision3d.utils.registration_utils import get_corr_indices
from utils.transform import Transform
import pandas as pd
from vision3d.utils.point_cloud_utils import pairwise_distance, apply_transform,axis_angle_to_rotation,rotation_to_axis_angle,get_point2trans_index

from torch_cluster import knn_graph

import pinocchio as pin
import quaternion


eps = 1e-8

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, required=True, default=24)
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'extra', 'val'], default='3DMatch')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--voxel_size', type=float, default=0.00055)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--snapshot_dir', type=str, default='')
    parser.add_argument('--logs_dir', type=str, default='')
    parser.add_argument('--proposals_dir', type=str, default='')
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
        training=True
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
    overlap_ratio_count=torch.zeros(10)
    overlap_ratio=torch.zeros(10)
    overlap_ratio_ecc=torch.zeros(10)
    frag_id = 0
    mean_pre=0
    mean_recall=0
    mean_pre_ori=0
    mean_recall_ori=0
    mean_time = 0
    count=0
    fail=0

    mean_mask_pre=torch.zeros(10).cuda()
    mean_mask_recall=torch.zeros(10).cuda()
    gt_mask_mean_precision=0
    mean_mask_iou=torch.zeros(10).cuda()
    mean_node_succ=torch.zeros(config.coarse_matching_num_proposal).cuda()
    id_list=[]
    count_sym=0
    estimated_transforms_scene = []
    estimated_transforms_ori_scene = []

    id_list=[]
    inlier_ratio_num=0
    print(num_iter_per_epoch)
    start=0
    a = time.time()
    for i, data_dict in enumerate(data_loader):
        # print(i)
        data_dict = to_cuda(data_dict)
        timer.add_prepare_time()
        with torch.no_grad():
            # breakpoint()
            ref_length_c = data_dict['stack_lengths'][-1][0].item()
            ref_length_m = data_dict['stack_lengths'][1][0].item()                
            points_c = data_dict['points'][-1].detach()
            points_m = data_dict['points'][1].detach()
                
            ref_points_c = points_c[:ref_length_c]
            src_points_c = points_c[ref_length_c:]

            ref_points_m = points_m[:ref_length_m]
            src_points_m = points_m[ref_length_m:]

            ref_length = data_dict['stack_lengths'][0][0].item()                
            points = data_dict['points'][0].detach()
                
            ref_points = points[:ref_length]
            src_points = points[ref_length:]
            try:
                if data_dict['scene_name'].item() == count_sym:
                    a = time.time()
                    output_dict = model(data_dict)
                    mean_time += time.time() - a
                    if output_dict['estimated_transforms'].shape[0]==0:
                        engine.logger.info(f'failed{i}')
                    estimated_transforms_scene.append(output_dict['estimated_transforms'])
                    estimated_transforms_ori_scene.append(output_dict['estimated_transforms_ori'])
                    inputs_dict = data_dict
                else:
                    print(count_sym)
                    count_sym = data_dict['scene_name'].item()

                    output_dict['estimated_transforms'] = torch.cat(estimated_transforms_scene, dim=0)
                    output_dict['estimated_transforms_ori'] = torch.cat(estimated_transforms_ori_scene, dim=0)
                    estimated_transforms_scene = []
                    estimated_transforms_ori_scene = []
                    mean_time += time.time() - a
                    result_dict = evaluator(output_dict, inputs_dict)
                    torch.cuda.empty_cache()

                    estimated_transforms=output_dict['estimated_transforms']
                    estimated_transforms_ori=output_dict['estimated_transforms_ori']
            
                    print(data_dict['scene_name'])
  

                    mean_pre+=result_dict['precision']
                    mean_recall+=result_dict['recall']
    

                    print('precision',100*mean_pre/(count+1))
                    print('recall',100*mean_recall/(count+1))    
                    
                    recall = result_dict['recall']
                    precision = result_dict['precision']
                    engine.logger.info(f'frag:{count_sym}')
                    engine.logger.info(f'precision:{precision}')
                    engine.logger.info(f'recall:{recall}')

                    recall_trans_index = result_dict['recall_trans']
                    valid_indices = recall_trans_index[recall_trans_index>=0]
                    transform_visulized = estimated_transforms[valid_indices].cpu().numpy()

                    count+=1 

                    src_R=pairwise_distance(src_points_m,src_points_m).max()
                    src_R=torch.sqrt(src_R)
                    start+=1
                    
                    output_dict = model(data_dict)
                    estimated_transforms_scene.append(output_dict['estimated_transforms'])
                    estimated_transforms_ori_scene.append(output_dict['estimated_transforms_ori'])
                    inputs_dict = data_dict
            except Exception as inst:
                start+=1
                fail+=1
            
    output_dict['estimated_transforms'] = torch.cat(estimated_transforms_scene, dim=0)
    output_dict['estimated_transforms_ori'] = torch.cat(estimated_transforms_ori_scene, dim=0)
    estimated_transforms_scene = []
    estimated_transforms_ori_scene = []
    mean_time += time.time() - a
    print(mean_time/(count+1))
    result_dict = evaluator(output_dict, inputs_dict)
    torch.cuda.empty_cache()
    estimated_transforms=output_dict['estimated_transforms']
    estimated_transforms_ori=output_dict['estimated_transforms_ori']

    print(data_dict['frag_id0'])
    print(len(estimated_transforms))
    print(len(estimated_transforms_ori))

    mean_pre+=result_dict['precision']
    mean_recall+=result_dict['recall']

    print('precision',100*mean_pre/(count+1))
    print('recall',100*mean_recall/(count+1))      
    print('F1_SCORE',2*(mean_pre/count)*(mean_recall/count)/((mean_recall/count)+(mean_pre/count)))
    print(count)


def main(rank=None, world_size=None):
    parser = make_parser()
    args = parser.parse_args()
    config.voxel_size = args.voxel_size
    config.snapshot_dir = args.snapshot_dir
    config.logs_dir = args.logs_dir
    config.process_robi_root = args.proposals_dir
    
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(local_rank=rank, world_size=world_size, log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        # engine.args.test_epoch=24
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

        test_loader, neighborhood_limits = Process_ROBI_test_data_loader(engine, config)
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

        # snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(engine.args.test_epoch))
        snapshot = './pretrain/robimatch.pth.tar'
        engine.load_snapshot(snapshot)

        start_time = time.time()
        run_one_epoch(
                engine, engine.args.test_epoch, test_loader, model, evaluator, training=False
            )
        

        loading_time = time.time() - start_time
        message = ' test_one_epoch: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)


if __name__ == '__main__':
    main()
