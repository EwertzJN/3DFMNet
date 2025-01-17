import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from vision3d.datasets.registration.threedmatch_kpconv_v1 import Process_Scan2cadKPConvDataset,Scan2cadKPConvDataset,get_dataloader
from vision3d.utils.point_cloud_utils import get_point_to_node_indices_and_masks,pairwise_distance,farthest_point_sample,get_knn_indices,get_compatibility,get_point2trans_index, get_node_nearest_centers_offset
from vision3d.utils.registration_utils import get_node_corr_indices_and_overlaps,to_o3d_pcd,to_array,to_tensor,cal_R
from sklearn import cluster
from vision3d.modules.geometry.functional import random_ball_query
from vision3d.utils.torch_utils import to_cuda, all_reduce_dict

from configs.config_scan2cad_focus import config

def generate_fine_level_data(data_dict, config, augment_center=False):
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

    _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = get_point_to_node_indices_and_masks(
            ref_points_m, ref_points_c, config.point_to_node_max_point
        )
    _, src_node_masks, src_node_knn_indices, src_node_knn_masks = get_point_to_node_indices_and_masks(
                src_points_m, src_points_c, config.point_to_node_max_point
            )
    
    
    sentinel_point = torch.zeros(1, 3).to(ref_points_c.device)
    # 因为ref_node_knn_indices是按照len(ref_points_m)填充，所以要把最后一维拓展开方便索引
    ref_padded_points_m = torch.cat([ref_points_m, sentinel_point], dim=0)
    src_padded_points_m = torch.cat([src_points_m, sentinel_point], dim=0)

    ref_node_knn_points = ref_padded_points_m[ref_node_knn_indices] # (n, k, 3)
    src_node_knn_points = src_padded_points_m[src_node_knn_indices]
    
    # 根据transform算超点的对应关系，然后计算knn领域的overlap(一个实例大概300-400对correspondeces)
    # breakpoint()
    gt_node_corr_indices, gt_node_corr_overlaps, gt_instance_centers = get_node_corr_indices_and_overlaps(
        ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform, config.ground_truth_positive_radius/2, config.ground_truth_positive_radius/2,
        ref_masks=ref_node_masks, src_masks=src_node_masks,
        ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
        )
    # gt_corr_trans_index 存储的是每一个correspondence的实例标签
    # gt_corr_trans_index_sum 用于存储每个实例的correspondence有多长

    gt_corr_trans_index=torch.full([len(gt_node_corr_indices)],0)
    gt_corr_trans_index_sum=torch.zeros(len(transforms))
    gt_corr_trans_index_sum[0]=len(gt_node_corr_indices)
    for i in range(len(transforms)-1):
        transform=transforms[i+1]
        gt_node_corr_indices_tmp, gt_node_corr_overlaps_tmp, gt_instance_centers_tmp = get_node_corr_indices_and_overlaps(
            ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform,config.ground_truth_positive_radius/2,config.ground_truth_positive_radius/2,
            ref_masks=ref_node_masks, src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
            )
        gt_corr_trans_index_tmp=torch.full([len(gt_node_corr_indices_tmp)],i+1)
        gt_instance_centers = torch.cat([gt_instance_centers, gt_instance_centers_tmp], dim=0)
        gt_node_corr_indices=torch.cat([gt_node_corr_indices,gt_node_corr_indices_tmp], dim=0)
        gt_node_corr_overlaps=torch.cat([gt_node_corr_overlaps,gt_node_corr_overlaps_tmp], dim=0)
        gt_corr_trans_index=torch.cat([gt_corr_trans_index,gt_corr_trans_index_tmp], dim=0)
        gt_corr_trans_index_sum[i+1]=len(gt_node_corr_indices_tmp)

    min_corrds, _ = torch.min(src_points_c, dim=0)
    max_corrds, _ = torch.max(src_points_c, dim=0)
    random_number = random.choice([1, 1.2, 1.5])
    offset_thresh = torch.norm(max_corrds-min_corrds)/2
    
    if augment_center:
        augment_gt_center_offset = torch.rand(gt_instance_centers.shape[0],3).to(gt_instance_centers.device) - 0.5
        gt_instance_centers = gt_instance_centers + augment_gt_center_offset * offset_thresh
    offset_thresh = offset_thresh*random_number

    point2trans_index=get_point2trans_index(ref_points_m,src_points_c,transforms,config.ground_truth_positive_radius*2)
    gt_instance_label = point2trans_index+1
    gt_instance_label = gt_instance_label.bool()
    # offset_thresh = torch.norm(src_points_c-)/2

    src_centers = torch.mean(src_points_c, dim=0, keepdim=True).repeat(gt_instance_centers.shape[0], 1) #先看看维度
    # random_number = random.choice([1, 1.2, 1.5])
    # offset_thresh = torch.max(torch.norm(src_points_c-torch.mean(src_points_c, dim=0, keepdim=True))) * random_number
    # offset_thresh = offset_thresh / 2
    ref_fine_points_indices = random_ball_query(points=ref_points_f.unsqueeze(0).transpose(2,1).contiguous(), centroids=gt_instance_centers.unsqueeze(0).transpose(2,1).contiguous(), num_sample=4096, radius=offset_thresh).squeeze(0)
    src_fine_points_indices = random_ball_query(points=src_points_f.unsqueeze(0).transpose(2,1).contiguous(), centroids=src_centers.unsqueeze(0).transpose(2,1).contiguous(), num_sample=4096, radius=offset_thresh).squeeze(0)
    ref_fine_points = ref_points_f[ref_fine_points_indices]
    src_fine_points = src_points_f[src_fine_points_indices]
    # breakpoint()
    # import open3d as o3d
    # ref_fine_pcd = o3d.geometry.PointCloud()
    # ref_fine_pcd.points = o3d.utility.Vector3dVector(ref_fine_points[0].squeeze(0).cpu().numpy())
    # o3d.io.write_point_cloud('exp.pcd', ref_fine_pcd)
    ref_fine_points_norm = ref_fine_points - gt_instance_centers.unsqueeze(1).repeat(1,ref_fine_points.shape[1],1)
    src_fine_points_norm = src_fine_points - src_centers.unsqueeze(1).repeat(1,src_fine_points.shape[1],1)     


    # ref_fine_feats, src_fine_feats, matching_scores = self.dcp(ref_fine_points_norm, src_fine_points_norm)
    gt_rot = transforms[:, :3, :3]  
    gt_trans = transforms[:, :3, 3] + (src_centers.unsqueeze(1)@gt_rot.transpose(-1,-2)).squeeze(1) - gt_instance_centers
    transform_norm = torch.eye(4).unsqueeze(0).repeat(transforms.shape[0], 1, 1).to(transforms.device)
    transform_norm[:, :3, :3] = gt_rot
    transform_norm[:, :3, 3] = gt_trans

    output_dict['ref_fine_points'] = ref_fine_points
    output_dict['src_fine_points'] = src_fine_points
    output_dict['ref_fine_points_norm'] = ref_fine_points_norm
    output_dict['src_fine_points_norm'] = src_fine_points_norm
    output_dict['transforms_norm'] = transform_norm
    output_dict['transforms'] = transforms
    output_dict['sym'] = sym

    return output_dict


if __name__ == '__main__':
    train_dataset = Process_Scan2cadKPConvDataset(config.process_scan2cad_root,'train',
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    train_data_loader, neighborhood_limits = get_dataloader(train_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=False,
                                                            sampler=None,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    # count = 4705
    count = 0
    for c_iter, inputs in enumerate(train_data_loader): 
        input_dict = to_cuda(inputs)
        result_dict = generate_fine_level_data(input_dict, config)
        for i in range(result_dict['ref_fine_points'].shape[0]):
            ref_fine_points = result_dict['ref_fine_points'][i,:,:]
            src_fine_points = result_dict['src_fine_points'][i,:,:]
            transform = result_dict['transforms'][i,:,:].unsqueeze(0)
            sym = result_dict['sym']

            save_ref_fine_points = ref_fine_points.cpu().numpy()
            save_src_fine_points = src_fine_points.cpu().numpy()
            save_transforms = transform.cpu().numpy()
            savename = '/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/scan2cad/scan2cad_last_random_4096/'+'data{:05d}.npz'.format(count)
            np.savez(savename, points0=save_ref_fine_points, points1=save_src_fine_points, trans=save_transforms, frag_id1=sym, scene_name=sym, sym=sym)
            count += 1
        # breakpoint()
        print(c_iter)
        # breakpoint()