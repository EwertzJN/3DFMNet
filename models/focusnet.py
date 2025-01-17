import math

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch_scatter import scatter_min, scatter_mean, scatter_max

from sklearn import cluster
from vision3d.modules.geometry.functional import random_ball_query
from vision3d.modules.kpconv.kpfcnn import make_kpfcnn_encoder, make_kpfcnn_decoder, KPEncoder, KPDecoder
from vision3d.utils.point_cloud_utils import get_point_to_node_indices_and_masks,pairwise_distance,farthest_point_sample,get_knn_indices,get_compatibility,get_point2trans_index, get_node_nearest_centers_offset
from vision3d.utils.registration_utils import get_node_corr_indices_and_overlaps,to_o3d_pcd,to_array,to_tensor,cal_R
from vision3d.utils.torch_utils import index_select
from .modules_robi import (
     CoarseMatching, FineMatching,  CoarseTargetGenerator, LearnableLogOptimalTransport,CoarseMaskTransformer2
)
import time
from queue import Queue
import os,copy
import open3d as o3d
#Scan2cad without sp


      
class FocusingNet(nn.Module):
    def __init__(self, config):
        super(FocusingNet, self).__init__()
        self.config = config
        self.final_feats_dim = config.final_feats_dim
        self.point_to_node_max_point = config.point_to_node_max_point
        self.with_slack = config.fine_matching_with_slack
        self.pos_radius = config.ground_truth_positive_radius
        self.point_pos_radius=config.fine_sinkhorn_loss_positive_radius
        self.dbscan_eps = config.dbscan_eps
        self.dbscan_min_samples = config.dbscan_min_samples
        # KPConv Encoder
        encoder_dict = make_kpfcnn_encoder(config, config.in_features_dim)
        self.encoder = KPEncoder(encoder_dict)

        # GNN part

        self.transformer = CoarseMaskTransformer2(
            encoder_dict['out_dim'],
            config.coarse_tfm_feats_dim,
            config.coarse_tfm_feats_dim,
            config.coarse_tfm_num_head,
            config.coarse_tfm_architecture,
            config.coarse_tfm_bin_size_d,
            config.coarse_tfm_bin_size_a,
            config.coarse_tfm_angle_k,
            config.max_neighboor,
            config.geodesic_radis
        )
        self.instance_mask_thre=config.instance_mask_thre
        # self.eval_iou=False

        # Instance predict head
        self.instance_proj = nn.Sequential(nn.Linear(256,256), nn.LayerNorm(256))
        self.instance_pred_head = nn.Sequential(
                                    nn.Linear(2*256, 256),nn.LayerNorm(256), nn.ReLU() , 
                                    nn.Linear(256, 1))

        # Center offset head 
        self.offset_head = nn.Sequential(nn.Linear(256, 128),nn.LayerNorm(128), nn.ReLU(),
                                         nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
                                         nn.Linear(64,3))

        # KPConv Decoder
        # decoder_dict = make_kpfcnn_decoder(config, encoder_dict, encoder_dict['out_dim'], config.final_feats_dim)
        # self.decoder = KPDecoder(decoder_dict)
        # self.finematch_max_point=config.finematch_max_point
        # self.max_ref_nodes=config.max_ref_nodes
        # self.max_neighboor=config.max_neighboor
        # self.max_sample_neighboor=config.max_sample_neighboor

    def predict_instance_score(self, ref_node_neighbor_feats, cross_position_embeddings):
        res_feats = self.instance_proj(ref_node_neighbor_feats)
        instance_feats = torch.cat((ref_node_neighbor_feats-res_feats,cross_position_embeddings.squeeze(1)),dim=-1)
        instance_feats = self.instance_pred_head(instance_feats)
        instance_local_feats, _ = torch.max(instance_feats, 1, keepdim=False)
        pred_instance_score = instance_local_feats#.sigmoid()
        return pred_instance_score
    
    def forward(self, data_dict):
        output_dict = {}
        feats_f = data_dict['features'].detach()
        ref_length_c = data_dict['stack_lengths'][-1][0].item()
        ref_length_m = data_dict['stack_lengths'][1][0].item()
        ref_length_f = data_dict['stack_lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_m = data_dict['points'][1].detach()
        points_f = data_dict['points'][0].detach()
        transforms= data_dict['transform'].detach()
        sym=data_dict['frag_id0']
        output_dict['sym'] = sym
        transform=transforms[0]
        
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_m = points_m[:ref_length_m]
        src_points_m = points_m[ref_length_m:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_m'] = ref_points_m
        output_dict['src_points_m'] = src_points_m
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f

        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = get_point_to_node_indices_and_masks(
                ref_points_m, ref_points_c,self.point_to_node_max_point
            )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = get_point_to_node_indices_and_masks(
                    src_points_m, src_points_c, self.point_to_node_max_point
                )
        
        sentinel_point = torch.zeros(1, 3).cuda()
        ref_padded_points_m = torch.cat([ref_points_m, sentinel_point], dim=0)
        src_padded_points_m = torch.cat([src_points_m, sentinel_point], dim=0)

        ref_node_knn_points = ref_padded_points_m[ref_node_knn_indices] # (n, k, 3)
        src_node_knn_points = src_padded_points_m[src_node_knn_indices]
        
        gt_node_corr_indices, gt_node_corr_overlaps, gt_instance_centers = get_node_corr_indices_and_overlaps(
            ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform,self.pos_radius/2,self.point_pos_radius/2,
            ref_masks=ref_node_masks, src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
            )

        gt_corr_trans_index=torch.full([len(gt_node_corr_indices)],0)
        gt_corr_trans_index_sum=torch.zeros(len(transforms))
        gt_corr_trans_index_sum[0]=len(gt_node_corr_indices)
        for i in range(len(transforms)-1):
            transform=transforms[i+1]
            gt_node_corr_indices_tmp, gt_node_corr_overlaps_tmp, gt_instance_centers_tmp = get_node_corr_indices_and_overlaps(
                ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform,self.pos_radius/2,self.point_pos_radius/2,
                ref_masks=ref_node_masks, src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
                )
            gt_corr_trans_index_tmp=torch.full([len(gt_node_corr_indices_tmp)],i+1)
            gt_instance_centers = torch.cat([gt_instance_centers, gt_instance_centers_tmp], dim=0)
            gt_node_corr_indices=torch.cat([gt_node_corr_indices,gt_node_corr_indices_tmp], dim=0)
            gt_node_corr_overlaps=torch.cat([gt_node_corr_overlaps,gt_node_corr_overlaps_tmp], dim=0)
            gt_corr_trans_index=torch.cat([gt_corr_trans_index,gt_corr_trans_index_tmp], dim=0)
            gt_corr_trans_index_sum[i+1]=len(gt_node_corr_indices_tmp)

        gt_corr_trans_index=gt_corr_trans_index.cuda()
        
        gt_center_offsets, gt_centers = get_node_nearest_centers_offset(ref_points_c, gt_instance_centers)
        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps
        # output_dict['gt_center_offsets'] = gt_center_offsets
        output_dict['gt_centers'] = gt_centers
        
        feats_c, skip_feats = self.encoder(feats_f, data_dict)

        # feats_m = self.decoder(feats_c, skip_feats, data_dict)

        # ref_feats_m = feats_m[:ref_length_m]
        # src_feats_m = feats_m[ref_length_m:]
        # output_dict['ref_feats_m'] = ref_feats_m
        # output_dict['src_feats_m'] = src_feats_m
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        """ start_time=time.time() """
        with  torch.no_grad():
            src_R=pairwise_distance(src_points_m,src_points_m).max()       
        if self.training:
            point2trans_index=get_point2trans_index(ref_points_c,src_points_c,transforms,self.pos_radius*4)
            ref_feats_node, src_feats_node ,ref_node_neighbor_indices,src_node_neighbor_indices,geo_dist,pred_masks_list,attn_masks_list, _, cross_position_embeddings= self.transformer(
                ref_points_c, src_points_c, ref_feats_c, src_feats_c,torch.sqrt(src_R)
            )
            ref_feats_node_norm = F.normalize(ref_feats_node, p=2, dim=1)
            src_feats_node_norm = F.normalize(src_feats_node, p=2, dim=1)

            output_dict['ref_feats_c'] = ref_feats_node_norm
            output_dict['src_feats_c'] = src_feats_node_norm

            gt_node_corr_trans_index=point2trans_index[ref_node_neighbor_indices]#(num_proposal, max_point)
            sample_proposal_tran_index=point2trans_index
            sample_proposal_tran_index=sample_proposal_tran_index.unsqueeze(1)
            sample_proposal_tran_index=sample_proposal_tran_index.expand(sample_proposal_tran_index.shape[-2],ref_node_neighbor_indices.shape[-1])
            gt_masks=torch.eq(gt_node_corr_trans_index,sample_proposal_tran_index)
                      
            # gt_center_offsets = get_node_nearest_centers_offset(ref_points_c[ref_node_corr_indices], gt_instance_centers)

            gt_instance_label = point2trans_index+1
            gt_instance_label = gt_instance_label.bool().float()
            ref_node_neighbor_feats = ref_feats_node_norm[ref_node_neighbor_indices]
            pred_instance_score = self.predict_instance_score(ref_node_neighbor_feats, cross_position_embeddings)
            output_dict['pred_instance_score'] = pred_instance_score
            output_dict['gt_instance_label'] = gt_instance_label

            output_dict['gt_center_offsets'] = gt_center_offsets[gt_instance_label.bool()]
            pred_center_offsets = self.offset_head(ref_feats_node_norm)
            output_dict['pred_center_offsets'] = pred_center_offsets[gt_instance_label.bool()]

        else :
            ref_feats_node, src_feats_node ,ref_node_neighbor_indices,src_node_neighbor_indices,geo_dist,pred_masks_list,attn_masks_list,mask_attention_score_list,cross_position_embeddings= self.transformer(
                ref_points_c, src_points_c, ref_feats_c, src_feats_c,torch.sqrt(src_R)
            )
            ref_feats_node_norm = F.normalize(ref_feats_node, p=2, dim=1)
            src_feats_node_norm = F.normalize(src_feats_node, p=2, dim=1)

            pred_center_offsets = self.offset_head(ref_feats_node_norm)

            output_dict['pred_center_offsets'] = pred_center_offsets
            output_dict['ref_feats_c'] = ref_feats_node_norm
            output_dict['src_feats_c'] = src_feats_node_norm

            ref_node_neighbor_feats = ref_feats_node_norm[ref_node_neighbor_indices]
            pred_instance_score = self.predict_instance_score(ref_node_neighbor_feats, cross_position_embeddings)
            output_dict['pred_instance_score'] = pred_instance_score
            
            with torch.no_grad():
                point2trans_index=get_point2trans_index(ref_points_c,src_points_c,transforms,self.pos_radius*2)
                gt_instance_label = point2trans_index+1
                gt_instance_label = gt_instance_label.bool().float()
                output_dict['gt_instance_label'] = gt_instance_label
                pred_instance_label = (pred_instance_score>0.5).bool().squeeze(1)
                min_corrds, _ = torch.min(src_points_c, dim=0)
                max_corrds, _ = torch.max(src_points_c, dim=0)
                offset_thresh = torch.norm(max_corrds-min_corrds)/2
                pred_point2trans_index = point2trans_index[pred_instance_label]
                unique_pred_index = torch.unique(pred_point2trans_index)
                if unique_pred_index[0] == -1:
                    pred_instance_num = unique_pred_index.shape[0]-1
                else:
                    pred_instance_num = unique_pred_index.shape[0]
                pred_noise_num = torch.eq(pred_point2trans_index,-1).sum().item()
                pred_instance_recall = pred_instance_num / transforms.shape[0]
                pred_offset_error = torch.mean(torch.norm((ref_points_c+pred_center_offsets-gt_centers),dim=-1)[pred_instance_label])
                pred_center = ref_points_c + pred_center_offsets
                output_dict['pred_points'] = ref_points_c[pred_instance_label]
                output_dict['pred_center'] = pred_center[pred_instance_label]
                output_dict['pred_instance_recall'] = pred_instance_recall
                output_dict['pred_offset_error'] = pred_offset_error
                output_dict['pred_noise_percent'] = pred_noise_num / pred_point2trans_index.shape[0]
                pred_center_cluster = pred_center[pred_instance_label].cpu().numpy()
                dbscan = cluster.DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
                dbscan.fit(pred_center_cluster)
                labels = dbscan.labels_
                labels = torch.from_numpy(labels)
                unique_labels = torch.unique(labels)
                mean_coordinates = []
                for label in unique_labels:
                    if label == -1:
                        continue
                    # breakpoint()
                    mask = labels == label
                    points_in_label = pred_center[pred_instance_label][mask.flatten()] 
                    mean_coordinate = torch.mean(points_in_label, dim=0, keepdim=True) 
                    mean_coordinates.append(mean_coordinate)
                pred_instance_centers = torch.cat(mean_coordinates, dim=0)
                output_dict['pred_instance_centers'] = pred_instance_centers
                
                pred_centers_dist = pairwise_distance(pred_instance_centers, gt_instance_centers)
                pred_instance_error, pred_instance_indices = torch.min(pred_centers_dist, dim=1)
                pred_instance_indices = pred_instance_indices[pred_instance_error<offset_thresh*0.1]
                pred_instance_label = torch.unique(pred_instance_indices)
                pred_cluster_instance_recall = pred_instance_label.shape[0] / gt_instance_centers.shape[0]
                pref_cluster_instance_precision = pred_instance_indices.shape[0] / pred_instance_centers.shape[0]
                output_dict['pred_cluster_instance_recall'] = pred_cluster_instance_recall
                output_dict['pred_cluster_instance_precision'] = pref_cluster_instance_precision

                random_number = random.choice([1, 1.2, 1.5])
                offset_radius = torch.norm(max_corrds-min_corrds)*random_number/2
                src_centers = torch.mean(src_points_c, dim=0, keepdim=True).repeat(pred_instance_centers.shape[0], 1) #先看看维度
                # breakpoint()
                ref_fine_points_indices = random_ball_query(points=ref_points_m.unsqueeze(0).transpose(2,1).contiguous(), centroids=pred_instance_centers.unsqueeze(0).transpose(2,1).contiguous(), num_sample=4096, radius=offset_radius).squeeze(0)
                src_fine_points_indices = random_ball_query(points=src_points_m.unsqueeze(0).transpose(2,1).contiguous(), centroids=src_centers.unsqueeze(0).transpose(2,1).contiguous(), num_sample=4096, radius=offset_radius).squeeze(0)
                ref_fine_points = ref_points_m[ref_fine_points_indices]
                src_fine_points = src_points_m[src_fine_points_indices]
                ref_fine_points_norm = ref_fine_points - pred_instance_centers.unsqueeze(1).repeat(1,ref_fine_points.shape[1],1)
                src_fine_points_norm = src_fine_points - src_centers.unsqueeze(1).repeat(1,src_fine_points.shape[1],1)
                
                output_dict['ref_fine_points'] = ref_fine_points
                output_dict['src_fine_points'] = src_fine_points
                output_dict['ref_fine_points_norm'] = ref_fine_points_norm
                output_dict['src_fine_points_norm'] = src_fine_points_norm
                
        return output_dict
        

def create_model(config):
    model = FocusingNet(config)
    return model

def generate_fine_level_data(data_dict, config):
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
    
    
    sentinel_point = torch.zeros(1, 3).cuda()
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

    # min_corrds, _ = torch.min(src_points_c, dim=0)
    # max_corrds, _ = torch.max(src_points_c, dim=0)
    # # random_number = random.choice([0.8, 1.2, 1.5])
    # # offset_thresh = torch.norm(max_corrds-min_corrds)*random_number/2
    # offset_thresh = torch.norm(max_corrds-min_corrds)*0.7/2

    src_centers = torch.mean(src_points_c, dim=0, keepdim=True).repeat(gt_instance_centers.shape[0], 1) #先看看维度
    offset_thresh = torch.max(torch.norm(src_points_c-torch.mean(src_points_c, dim=0, keepdim=True)))
    # offset_thresh = offset_thresh / 2
    breakpoint()
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points_f.cpu().numpy())
    ref_pcd.voxel_down_sample(voxel_size=0.00055)
    ref_points_save = torch.from_numpy(np.asarray(ref_pcd.points)).to(device=ref_points_f.device)
    
    src_pcd = o3d.geometry.PointCloud() 
    src_pcd.points = o3d.utility.Vector3dVector(src_points_f.cpu().numpy())
    src_pcd.voxel_down_sample(voxel_size=0.00055)
    src_points_save = torch.from_numpy(np.asarray(src_pcd.points)).to(device=ref_points_f.device)
    ref_fine_points_indices = random_ball_query(points=ref_points_save.unsqueeze(0).transpose(2,1).contiguous(), centroids=gt_instance_centers.unsqueeze(0).transpose(2,1).contiguous(), num_sample=1024, radius=offset_thresh).squeeze(0)
    src_fine_points_indices = random_ball_query(points=src_points_save.unsqueeze(0).transpose(2,1).contiguous(), centroids=src_centers.unsqueeze(0).transpose(2,1).contiguous(), num_sample=1024, radius=offset_thresh).squeeze(0)
    ref_fine_points = ref_points_save[ref_fine_points_indices]
    src_fine_points = src_points_save[src_fine_points_indices]
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

    return output_dict

def main():
    from configs.config_robi_focus import config
    model = create_model(config)
    print(model)


if __name__ == '__main__':
    main()
