import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from vision3d.datasets.registration.threedmatch_kpconv_v1 import Process_Scan2cadKPConvDataset,ROBISENCEDataset,get_dataloader
from vision3d.utils.point_cloud_utils import get_point_to_node_indices_and_masks,pairwise_distance,farthest_point_sample,get_knn_indices,get_compatibility,get_point2trans_index, get_node_nearest_centers_offset,apply_transform
from vision3d.utils.registration_utils import get_node_corr_indices_and_overlaps,to_o3d_pcd,to_array,to_tensor,cal_R
from sklearn import cluster
from vision3d.modules.geometry.functional import random_ball_query
from vision3d.utils.torch_utils import to_cuda, all_reduce_dict
import open3d as o3d
import argparse
import os
from configs.config_robi_focus import config

def generate_fine_level_data(data_dict, config, pred_center=None):
    output_dict = {}
    ref_length_c = data_dict['stack_lengths'][-1][0].item()
    ref_length_m = data_dict['stack_lengths'][1][0].item()
    ref_length_f = data_dict['stack_lengths'][0][0].item()
    points_c = data_dict['points'][-1].detach()
    points_m = data_dict['points'][1].detach()
    points_f = data_dict['points'][0].detach()
    transforms= data_dict['transform'].detach()
    sym=data_dict['frag_id0']
    transform=transforms[0]
    
    ref_points_c = points_c[:ref_length_c]
    src_points_c = points_c[ref_length_c:]
    ref_points_m = points_m[:ref_length_m]
    src_points_m = points_m[ref_length_m:]
    ref_points_f = points_f[:ref_length_f]
    src_points_f = points_f[ref_length_f:]
   
    src_node_centers = torch.mean(src_points_c, dim=0).unsqueeze(0)

    gt_instance_centers = apply_transform(src_node_centers, transform).squeeze(1)
    # breakpoint()
    if pred_center is None:
        for i in range(len(transforms)-1):
            transform=transforms[i+1]
            gt_instance_centers_tmp = apply_transform(src_node_centers, transform).squeeze(1)
            gt_instance_centers = torch.cat([gt_instance_centers, gt_instance_centers_tmp], dim=0)
    # breakpoint()
    else:
        pred_centers = pred_center.to(ref_points_c.device)
        # pred_centers_dist, _ = pairwise_distance(pred_centers, pred_centers)
        
        gt_instance_centers = pred_center.to(ref_points_c.device)

    min_corrds, _ = torch.min(src_points_c, dim=0)
    max_corrds, _ = torch.max(src_points_c, dim=0)
    # random_number = random.choice([1, 1.2, 1.5])
    offset_thresh = torch.norm(max_corrds-min_corrds)*1.5/2
    # offset_thresh = torch.norm(src_points_c-)/2

    src_centers = torch.mean(src_points_c, dim=0, keepdim=True).repeat(gt_instance_centers.shape[0], 1) 
    # breakpoint()
    ref_fine_points_indices = random_ball_query(points=ref_points_f.unsqueeze(0).transpose(2,1).contiguous(), 
                                                centroids=gt_instance_centers.unsqueeze(0).transpose(2,1).contiguous(), 
                                                num_sample=8192, 
                                                radius=offset_thresh).squeeze(0)
    src_fine_points_indices = random_ball_query(points=src_points_f.unsqueeze(0).transpose(2,1).contiguous(), 
                                                centroids=src_centers.unsqueeze(0).transpose(2,1).contiguous(), 
                                                num_sample=8192, 
                                                radius=offset_thresh).squeeze(0)
    ref_fine_points = ref_points_f[ref_fine_points_indices]
    src_fine_points = src_points_f[src_fine_points_indices]

    output_dict['ref_fine_points'] = ref_fine_points
    output_dict['src_fine_points'] = src_fine_points
    output_dict['transforms'] = transforms
    output_dict['sym'] = sym

    return output_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detcenter_dir', type=str, default='')
    parser.add_argument('--proposal_dir', type=str, default='/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/robi/gt_proposals')
    args = parser.parse_args()
    train_dataset = ROBISENCEDataset(0.0015,config.robi_root, 'train',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    # train_sampler = torch.utils.data.DistributedSampler(train_dataset) if engine.distributed else None
    train_data_loader, neighborhood_limits = get_dataloader(train_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=False,
                                                            sampler=None,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    
    test_dataset = ROBISENCEDataset(0.0001,config.robi_root, 'test',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=False,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    
    test_data_loader, neighborhood_limits = get_dataloader(test_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=False,
                                                            sampler=None,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    # count = 46628
    count = 0
    for c_iter, inputs in enumerate(train_data_loader): 
        input_dict = to_cuda(inputs)
        center_file_name = os.path.join(args.detcenter_dir, 'data{:05}.npz'.format(c_iter))
        center_file = np.load(center_file_name)
        pred_center = center_file['pred_centers']
        pred_center_tensor = torch.from_numpy(pred_center)
        result_dict = generate_fine_level_data(input_dict, config, pred_center=pred_center_tensor)
        for i in range(result_dict['ref_fine_points'].shape[0]):
            ref_fine_points = result_dict['ref_fine_points'][i,:,:]
            src_fine_points = result_dict['src_fine_points'][i,:,:]
            transform = result_dict['transforms']
            sym = result_dict['sym']

            save_ref_fine_points = ref_fine_points.cpu().numpy()
            save_src_fine_points = src_fine_points.cpu().numpy()
            save_transforms = transform.cpu().numpy()
            savename = os.path.join(args.proposal_dir, 'data{:05d}.npz'.format(count))
            np.savez(savename, points0=save_ref_fine_points, points1=save_src_fine_points, trans=save_transforms, frag_id1=sym, scene_name=c_iter, sym=sym)
            count += 1
        # object_count = []
        # for i in range(input_dict['transform'].shape[0]):
        #     object_count.append(count)
        #     count += 1
        # # breakpoint()
        # object_count_np = np.array(object_count)
        # savename = '/data1/huile/zhangliyuan/multi_instance/detection_robi/gt_detection_count/' + 'count{:05d}.txt'.format(c_iter+2912)
        # np.savetxt(savename, object_count_np, fmt='%d')
        print(c_iter)
        # breakpoint()