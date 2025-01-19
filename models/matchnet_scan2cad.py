'''
 Copyright (c) 2024 Liyuan Zhang
 This code is based on Zhiyuan Yu's work on multi-instance point cloud registration
 Original code can be found at: https://github.com/zhiyuanYU134/MIRETR
'''

import math

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch_scatter import scatter_min, scatter_mean, scatter_max

from vision3d.modules.kpconv.kpfcnn import make_kpfcnn_encoder, make_kpfcnn_decoder, KPEncoder, KPDecoder
from vision3d.utils.point_cloud_utils import get_point_to_node_indices_and_masks,pairwise_distance,farthest_point_sample,get_knn_indices,get_compatibility,get_point2trans_index
from vision3d.utils.registration_utils import get_node_corr_indices_and_overlaps,to_o3d_pcd,to_array,to_tensor,cal_R
from vision3d.utils.torch_utils import index_select
from .modules_scan2cad import (
     CoarseMatching, FineMatching, CoarseTargetGenerator, LearnableLogOptimalTransport,CoarseMaskTransformer2
)
import time
from queue import Queue
import os,copy
import open3d as o3d
#Scan2cad without sp

      
class MatchNet(nn.Module):
    def __init__(self, config):
        super(MatchNet, self).__init__()
        self.final_feats_dim = config.final_feats_dim
        self.point_to_node_max_point = config.point_to_node_max_point
        self.with_slack = config.fine_matching_with_slack
        self.pos_radius = config.ground_truth_positive_radius
        self.point_pos_radius=config.fine_sinkhorn_loss_positive_radius
        # KPConv Encoder
        encoder_dict = make_kpfcnn_encoder(config, config.in_features_dim)
        self.encoder = KPEncoder(encoder_dict)
        #self.training=True

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
        self.eval_iou=False

        # Overlap predict head
        self.overlap_proj = nn.Sequential(nn.Linear(256,256), nn.LayerNorm(256))
        self.overlap_pred_head = nn.Sequential(
                                    nn.Linear(256, 128),nn.LayerNorm(128), nn.ReLU() , 
                                    nn.Linear(128, 1))
        
        # KPConv Decoder
        decoder_dict = make_kpfcnn_decoder(config, encoder_dict, encoder_dict['out_dim'], config.final_feats_dim)
        self.decoder = KPDecoder(decoder_dict)
        self.finematch_max_point=config.finematch_max_point
        self.max_ref_nodes=config.max_ref_nodes
        self.max_neighboor=config.max_neighboor
        self.max_sample_neighboor=config.max_sample_neighboor

        # Optimal Transport
        self.optimal_transport = LearnableLogOptimalTransport(config.sinkhorn_num_iter)

        # Correspondence Generator
        self.coarse_matching = CoarseMatching(
            config.coarse_matching_num_proposal,
            dual_softmax=config.coarse_matching_dual_softmax
        )

        self.fine_matching = FineMatching(
            config.cluster_thre,
            config.cluster_refine,
            config.fine_matching_max_num_corr,
            config.fine_matching_topk,
            mutual=config.fine_matching_mutual,
            with_slack=config.fine_matching_with_slack,
            threshold=config.fine_matching_confidence_threshold,
            conditional=config.fine_matching_conditional_score,
            matching_radius=config.fine_matching_positive_radius,
            min_num_corr=config.fine_matching_min_num_corr,
            num_registration_iter=config.fine_matching_num_registration_iter,
            
        )

        # Target Generator
        self.coarse_target = CoarseTargetGenerator(
            config.coarse_matching_num_target,
            overlap_thresh=config.coarse_matching_overlap_thresh
        )
    def Fine_Match_data_prepare(self,ref_node_neighbor_mask,ref_seed_neighbor_indices,
        ref_node_knn_masks,
        ref_node_knn_points,
        ref_node_knn_indices,
        ref_feats_m,
        ):
        
        num_proposal, max_point=ref_seed_neighbor_indices.shape
        K=ref_node_knn_masks.shape[-1]
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_seed_neighbor_indices].reshape(num_proposal,max_point*K) # (num_proposal, max_point*K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_seed_neighbor_indices] .reshape(num_proposal,max_point*K,-1) # (num_proposal, max_point*K,3)
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_seed_neighbor_indices].reshape(num_proposal,max_point*K)  # (num_proposal, max_point*K)
        
        sentinel_feat = torch.zeros(1, self.final_feats_dim).cuda()
        ref_padded_feats_m = torch.cat([ref_feats_m, sentinel_feat], dim=0)

        ref_node_corr_knn_feats = ref_padded_feats_m[ref_node_corr_knn_indices]  # (num_proposal, max_point*K, C)
        if ref_node_neighbor_mask is not None:
            ref_node_neighbor_mask=ref_node_neighbor_mask.unsqueeze(-1).expand(ref_node_neighbor_mask.shape[0], ref_node_neighbor_mask.shape[1],ref_node_knn_masks.shape[1])# (num_proposal, max_point, K)
            ref_node_neighbor_mask=ref_node_neighbor_mask.reshape(num_proposal,max_point*K) 
            ref_node_corr_knn_masks=torch.logical_and(ref_node_corr_knn_masks,ref_node_neighbor_mask)
        proposal_indices, ref_indices = torch.nonzero(ref_node_corr_knn_masks, as_tuple=True)
        all_ref_node_corr_points = ref_node_corr_knn_points[proposal_indices, ref_indices]
        all_ref_node_corr_feats = ref_node_corr_knn_feats[proposal_indices, ref_indices]

        unique_masks = torch.ne(proposal_indices[1:], proposal_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [proposal_indices.shape[0]]

        chunks = [(x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:])]
        num_proposal = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        indices_length=torch.zeros(num_proposal)
        indices_list=[]
        for  i, (x, y) in enumerate(chunks):
            if y-x>self.finematch_max_point:
                indices_length[i]=self.finematch_max_point
                indices_tmp=torch.arange(0, self.finematch_max_point,dtype=torch.float32)/self.finematch_max_point
                indices_tmp=(x+torch.floor(indices_tmp*(y-x))).to(torch.long)
                #indices_list.append(torch.arange(x, x+self.finematch_max_point,dtype=torch.long))
                indices_list.append(indices_tmp)
            else:
                indices_length[i]=y-x
                indices_list.append(torch.arange(x, y,dtype=torch.long))
        indices=torch.cat(indices_list,dim=0).cuda()

        stacked_ref_corr_points = index_select(all_ref_node_corr_points, indices, dim=0)  # (total, 3)
        stacked_ref_corr_feats = index_select(all_ref_node_corr_feats, indices, dim=0)  # (total, 3)
        stacked_ref_corr_masks = torch.full((len(stacked_ref_corr_points),1),True).cuda()  # (total)

        max_corr = self.finematch_max_point
        target_chunks = [(i * max_corr, i * max_corr + indices_length[i]) for i in range(num_proposal)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3).long()  # (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).long().cuda()  # (total, 3)

        local_ref_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
        local_ref_corr_points.index_put_([indices0, indices1], stacked_ref_corr_points)
        local_ref_corr_points = local_ref_corr_points.view(num_proposal, max_corr, 3)

        indices0 = indices.unsqueeze(1).expand(indices.shape[0], self.final_feats_dim).long()  # (total, 3)
        indices1 = torch.arange(self.final_feats_dim).unsqueeze(0).expand(indices.shape[0], self.final_feats_dim).long().cuda()  # (total, 3)

        local_ref_corr_feats = torch.zeros(num_proposal * max_corr, self.final_feats_dim).cuda()
        local_ref_corr_feats.index_put_([indices0, indices1], stacked_ref_corr_feats)
        local_ref_corr_feats = local_ref_corr_feats.view(num_proposal, max_corr, self.final_feats_dim)

        indices0 = indices.unsqueeze(1).long()  # (total, 3)
        indices1 = torch.arange(1).unsqueeze(0).expand(indices.shape[0], 1).long().cuda()  # (total, 3)

        local_ref_corr_masks = torch.full((num_proposal * max_corr,1),False).cuda()
        local_ref_corr_masks.index_put_([indices0,indices1], stacked_ref_corr_masks)
        local_ref_corr_masks = local_ref_corr_masks.view(num_proposal, max_corr)

        return local_ref_corr_points,local_ref_corr_feats,local_ref_corr_masks

    def predict_overlap_score(self, ref_node_neighbor_feats):
        res_feats = self.overlap_proj(ref_node_neighbor_feats)
        overlap_feats = ref_node_neighbor_feats-res_feats
        overlap_feats = self.overlap_pred_head(overlap_feats)
        instance_local_overlap_feats, _ = torch.max(overlap_feats, 1, keepdim=False)
        pred_overlap_score = instance_local_overlap_feats#.sigmoid()
        return pred_overlap_score
    
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
        """ print(src_points_m.shape)
        print(src_points_c.shape) """
        

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = get_point_to_node_indices_and_masks(
                ref_points_m, ref_points_c,self.point_to_node_max_point
            )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = get_point_to_node_indices_and_masks(
                    src_points_m, src_points_c, self.point_to_node_max_point
                )
        
        
        sentinel_point = torch.zeros(1, 3).cuda()
        ref_padded_points_m = torch.cat([ref_points_m, sentinel_point], dim=0)
        src_padded_points_m = torch.cat([src_points_m, sentinel_point], dim=0)

        ref_node_knn_points = ref_padded_points_m[ref_node_knn_indices]
        src_node_knn_points = src_padded_points_m[src_node_knn_indices]
        # breakpoint()
        gt_node_corr_indices, gt_node_corr_overlaps, _ = get_node_corr_indices_and_overlaps(
            ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform,self.pos_radius/2,self.point_pos_radius/2,
            ref_masks=ref_node_masks, src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
            )
        gt_corr_trans_index=torch.full([len(gt_node_corr_indices)],0)
        gt_corr_trans_index_sum=torch.zeros(len(transforms))
        gt_corr_trans_index_sum[0]=len(gt_node_corr_indices)
        for i in range(len(transforms)-1):
            transform=transforms[i+1]
            gt_node_corr_indices_tmp, gt_node_corr_overlaps_tmp, _ = get_node_corr_indices_and_overlaps(
                ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform,self.pos_radius/2,self.point_pos_radius/2,
                ref_masks=ref_node_masks, src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
                )
            gt_corr_trans_index_tmp=torch.full([len(gt_node_corr_indices_tmp)],i+1)
            gt_node_corr_indices=torch.cat([gt_node_corr_indices,gt_node_corr_indices_tmp], dim=0)
            gt_node_corr_overlaps=torch.cat([gt_node_corr_overlaps,gt_node_corr_overlaps_tmp], dim=0)
            gt_corr_trans_index=torch.cat([gt_corr_trans_index,gt_corr_trans_index_tmp], dim=0)
            gt_corr_trans_index_sum[i+1]=len(gt_node_corr_indices_tmp)
        """ gt_node_corr_indices_key,index,couts=torch.unique(gt_node_corr_indices, return_inverse=True,return_counts=True,dim=0)
        gt_node_corr_overlaps_new=torch.zeros(len(gt_node_corr_indices_key)).cuda()
        for i in range(len(index)):
            gt_node_corr_overlaps_new[index[i]]=gt_node_corr_overlaps[i] """

        gt_corr_trans_index=gt_corr_trans_index.cuda()
        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        
        # 2. KPFCNN Encoder
        feats_c, skip_feats = self.encoder(feats_f, data_dict)

        # 5. KPFCNN Decoder
        feats_m = self.decoder(feats_c, skip_feats, data_dict)


        # 5. Head for fine level matching
        ref_feats_m = feats_m[:ref_length_m]
        src_feats_m = feats_m[ref_length_m:]
        output_dict['ref_feats_m'] = ref_feats_m
        output_dict['src_feats_m'] = src_feats_m
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        """ start_time=time.time() """
        with  torch.no_grad():
            src_R=pairwise_distance(src_points_m,src_points_m).max()        
        if self.training:
            # 3. Conditional Transformer
            point2trans_index=get_point2trans_index(ref_points_c,src_points_c,transforms,self.pos_radius*4)
            ref_feats_node, src_feats_node ,ref_node_neighbor_indices,src_node_neighbor_indices,geo_dist,pred_masks_list,attn_masks_list, _= self.transformer(
                ref_points_c, src_points_c, ref_feats_c, src_feats_c,torch.sqrt(src_R)
            )
            """ loading_time = time.time() - start_time
            print("local_mask_transformer",loading_time)
            start_time=time.time() """
            # 4. Head for coarse level matching
            ref_feats_node_norm = F.normalize(ref_feats_node, p=2, dim=1)
            src_feats_node_norm = F.normalize(src_feats_node, p=2, dim=1)

            output_dict['ref_feats_c'] = ref_feats_node_norm
            output_dict['src_feats_c'] = src_feats_node_norm
            gt_node_corr_trans_index=point2trans_index[ref_node_neighbor_indices]#(num_proposal, max_point)
            sample_proposal_tran_index=point2trans_index
            sample_proposal_tran_index=sample_proposal_tran_index.unsqueeze(1)
            sample_proposal_tran_index=sample_proposal_tran_index.expand(sample_proposal_tran_index.shape[-2],ref_node_neighbor_indices.shape[-1])
            gt_masks=torch.eq(gt_node_corr_trans_index,sample_proposal_tran_index)

            # compute overlap mask
            gt_instance_label = point2trans_index+1
            gt_node_knn_instance_label = gt_instance_label[ref_node_neighbor_indices].bool()
            gt_instance_label = gt_instance_label.bool().float()
            ref_node_neighbor_feats = ref_feats_node_norm[ref_node_neighbor_indices]
            pred_overlap_score = self.predict_overlap_score(ref_node_neighbor_feats)
            output_dict['pred_instance_score'] = pred_overlap_score
            output_dict['gt_instance_label'] = gt_instance_label
            
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores,sel_indices,gt_corr_masks = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )#num_proposal

            ref_seed_neighbor_indices=ref_node_neighbor_indices[ref_node_corr_indices]
            src_seed_neighbor_indices=src_node_neighbor_indices[src_node_corr_indices]# (num_proposal, max_point)

            ref_node_neighbor_mask = gt_masks[ref_node_corr_indices]# (num_proposal, max_point)
            src_node_neighbor_num=ref_node_neighbor_mask.sum(1)
            src_node_neighbor_mask=torch.zeros_like(src_seed_neighbor_indices).cuda().bool()
            for i in range(len(src_node_neighbor_mask)):
                src_node_neighbor_mask[i][:src_node_neighbor_num[i]]=True

            output_dict['pred_masks_list']=pred_masks_list
            output_dict['attn_masks_list']=attn_masks_list
            output_dict['gt_masks']=gt_masks

            ref_node_neighbor_overlap_mask = gt_node_knn_instance_label[ref_node_corr_indices]# (num_proposal, max_point)
            ref_node_neighbor_overlap_mask = ref_node_neighbor_overlap_mask.unsqueeze(-1).expand(ref_node_neighbor_overlap_mask.shape[0], ref_node_neighbor_overlap_mask.shape[1],ref_node_knn_masks.shape[1])
            
        else :
            ref_feats_node, src_feats_node ,ref_node_neighbor_indices,src_node_neighbor_indices,geo_dist,pred_masks_list,attn_masks_list,mask_attention_score_list= self.transformer(
                ref_points_c, src_points_c, ref_feats_c, src_feats_c,torch.sqrt(src_R)
            )
            """ loading_time = time.time() - start_time
            print("local_mask_transformer",loading_time)
            start_time=time.time() """
            # 4. Head for coarse level matching
            ref_feats_node_norm = F.normalize(ref_feats_node, p=2, dim=1)
            src_feats_node_norm = F.normalize(src_feats_node, p=2, dim=1)

            output_dict['ref_feats_c'] = ref_feats_node_norm
            output_dict['src_feats_c'] = src_feats_node_norm
            with torch.no_grad():
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores,all_ref_corr_indices = self.coarse_matching(
                    ref_feats_node_norm, src_feats_node_norm
                )
                ref_seed_neighbor_indices=ref_node_neighbor_indices[ref_node_corr_indices]
                src_seed_neighbor_indices=src_node_neighbor_indices[src_node_corr_indices]# (num_proposal, max_point)

                # predict instance mask
                ref_node_neighbor_feats = ref_feats_node_norm[ref_node_neighbor_indices]
                pred_instance_score = self.predict_overlap_score(ref_node_neighbor_feats)
                pred_instance_label = pred_instance_score.sigmoid() > self.instance_mask_thre
                ref_node_neighbor_overlap_mask = pred_instance_label[ref_seed_neighbor_indices].squeeze(-1)
                ref_node_neighbor_overlap_mask = ref_node_neighbor_overlap_mask.unsqueeze(-1).expand(ref_node_neighbor_overlap_mask.shape[0], ref_node_neighbor_overlap_mask.shape[1],ref_node_knn_masks.shape[1])
                
                ref_node_neighbor_mask=pred_masks_list[-1]
                ref_node_neighbor_mask = (ref_node_neighbor_mask> self.instance_mask_thre).bool()# (num_proposal, max_point)
                ref_node_neighbor_mask=ref_node_neighbor_mask[ref_node_corr_indices]
                src_node_neighbor_num=ref_node_neighbor_mask.sum(1)
                src_node_neighbor_mask=torch.zeros_like(src_seed_neighbor_indices).cuda().bool()
                for i in range(len(src_node_neighbor_mask)):
                    src_node_neighbor_mask[i][:src_node_neighbor_num[i]]=True
                output_dict['ref_node_knn_points']=ref_node_knn_points
                output_dict['src_node_knn_points']=src_node_knn_points
                output_dict['node_knn_indices']=ref_seed_neighbor_indices
                output_dict['pred_masks_list']=pred_masks_list
                output_dict['attn_masks_list']=attn_masks_list
                output_dict['geo_dist']=geo_dist
                output_dict['mask_attention_score_list']=mask_attention_score_list


        """ start_time=time.time() """
            
        output_dict['ref_node_corr_indices']=ref_node_corr_indices
        output_dict['src_node_corr_indices']=src_node_corr_indices
        output_dict['ref_node_neighbor_indices']=ref_node_neighbor_indices
        output_dict['src_node_neighbor_indices']=src_node_neighbor_indices
        output_dict['ref_node_corr_indices']=ref_node_corr_indices
        output_dict['src_node_corr_indices']=src_node_corr_indices
        output_dict['ref_node_neighbor_indices']=ref_node_neighbor_indices
        output_dict['src_node_neighbor_indices']=src_node_neighbor_indices

        """ ref_neighbor_corr_knn_points,ref_neighbor_corr_knn_feats,ref_neighbor_knn_masks=self.Fine_Match_data_prepare(ref_node_neighbor_mask,ref_seed_neighbor_indices,
        ref_node_knn_masks,
        ref_node_knn_points,
        ref_node_knn_indices,
        ref_feats_m)

        src_neighbor_corr_knn_points,src_neighbor_corr_knn_feats,src_neighbor_knn_masks=self.Fine_Match_data_prepare(src_node_neighbor_mask,src_seed_neighbor_indices,
        src_node_knn_masks,
        src_node_knn_points,
        src_node_knn_indices,
        src_feats_m) """

        ref_node_neighbor_mask=ref_node_neighbor_mask.unsqueeze(-1).expand(ref_node_neighbor_mask.shape[0], ref_node_neighbor_mask.shape[1],ref_node_knn_masks.shape[1])# (num_proposal, max_point, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_seed_neighbor_indices]  # (num_proposal, max_point, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_seed_neighbor_indices]  # (num_proposal, max_point, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_seed_neighbor_indices]  # (num_proposal, max_point, K,3)
        src_node_corr_knn_points = src_node_knn_points[src_seed_neighbor_indices]  # (num_proposal, max_point, K,3)
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_seed_neighbor_indices]  # (num_proposal, max_point, K)
        src_node_corr_knn_indices= src_node_knn_indices[src_seed_neighbor_indices]  # (num_proposal, max_point, K)
        
        
        ref_neighbor_corr_knn_feats = torch.zeros(len(ref_node_neighbor_mask),self.finematch_max_point, self.final_feats_dim).cuda()
        src_neighbor_corr_knn_feats = torch.zeros(len(ref_node_neighbor_mask),self.finematch_max_point, self.final_feats_dim).cuda()
        ref_neighbor_corr_knn_points = torch.zeros(len(ref_node_neighbor_mask),self.finematch_max_point, 3).cuda()
        src_neighbor_corr_knn_points = torch.zeros(len(ref_node_neighbor_mask),self.finematch_max_point, 3).cuda()
        ref_neighbor_knn_masks=torch.full((len(ref_node_neighbor_mask),self.finematch_max_point),False).cuda()
        src_neighbor_knn_masks=torch.full((len(ref_node_neighbor_mask),self.finematch_max_point),False).cuda()

        sentinel_feat = torch.zeros(1, self.final_feats_dim).cuda()
        ref_padded_feats_m = torch.cat([ref_feats_m, sentinel_feat], dim=0)
        src_padded_feats_m = torch.cat([src_feats_m, sentinel_feat], dim=0)

        ref_node_corr_knn_feats = ref_padded_feats_m[ref_node_corr_knn_indices]  # (num_proposal, max_point, K, C)
        src_node_corr_knn_feats = src_padded_feats_m[src_node_corr_knn_indices]  # (num_proposal, max_point, K, C)

        ref_node_corr_knn_masks=torch.logical_and(ref_node_corr_knn_masks,ref_node_neighbor_mask)
        ref_node_corr_knn_masks=torch.logical_and(ref_node_corr_knn_masks, ref_node_neighbor_overlap_mask)
        
        for i in range(len(ref_node_neighbor_mask)):
            ref_tmp_points=ref_node_corr_knn_points[i]
            ref_tmp_masks=ref_node_corr_knn_masks[i]
            ref_tmp_feats=ref_node_corr_knn_feats[i]
            src_tmp_points=src_node_corr_knn_points[i][:src_node_neighbor_num[i]]
            src_tmp_masks=src_node_corr_knn_masks[i][:src_node_neighbor_num[i]]
            src_tmp_feats=src_node_corr_knn_feats[i][:src_node_neighbor_num[i]]

            ref_tmp_points=ref_tmp_points.reshape(-1,ref_tmp_points.shape[-1])
            src_tmp_points=src_tmp_points.reshape(-1,src_tmp_points.shape[-1])
            ref_tmp_feats=ref_tmp_feats.reshape(-1,ref_tmp_feats.shape[-1])
            src_tmp_feats=src_tmp_feats.reshape(-1,src_tmp_feats.shape[-1])

            ref_tmp_points=ref_tmp_points[ref_tmp_masks.reshape(-1)]
            src_tmp_points=src_tmp_points[src_tmp_masks.reshape(-1)]
            ref_tmp_feats=ref_tmp_feats[ref_tmp_masks.reshape(-1)]
            src_tmp_feats=src_tmp_feats[src_tmp_masks.reshape(-1)]

            if len(ref_tmp_points)>=self.finematch_max_point:
                inds = torch.LongTensor(random.sample(range(len(ref_tmp_points)), self.finematch_max_point)).cuda()
                #inds = np.random.choice(range(len(ref_tmp_points)), self.finematch_max_point, replace=False)
                ref_neighbor_corr_knn_feats[i] =ref_tmp_feats[inds] 
                ref_neighbor_corr_knn_points[i]=ref_tmp_points[inds] 
                ref_neighbor_knn_masks[i]=True
            else:
                ref_neighbor_corr_knn_feats[i][:len(ref_tmp_points)]=ref_tmp_feats
                ref_neighbor_corr_knn_points[i][:len(ref_tmp_points)]=ref_tmp_points
                ref_neighbor_knn_masks[i][:len(ref_tmp_points)]=True
            if len(src_tmp_points)>=self.finematch_max_point:
                inds = torch.LongTensor(random.sample(range(len(src_tmp_points)), self.finematch_max_point)).cuda()
                src_neighbor_corr_knn_feats[i] =src_tmp_feats[inds] 
                src_neighbor_corr_knn_points[i]=src_tmp_points[inds] 
                src_neighbor_knn_masks[i]=True
            else:
                src_neighbor_corr_knn_feats[i][:len(src_tmp_points)]=src_tmp_feats
                src_neighbor_corr_knn_points[i][:len(src_tmp_points)]=src_tmp_points
                src_neighbor_knn_masks[i][:len(src_tmp_points)]=True
        """ loading_time = time.time() - start_time
        print("point_match_prepare",loading_time)
        start_time=time.time() """

        # 8. Optimal transport
       # print(ref_neighbor_corr_knn_feats.shape)
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_neighbor_corr_knn_feats, src_neighbor_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / self.final_feats_dim ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_neighbor_knn_masks, src_neighbor_knn_masks)

        output_dict['matching_scores'] = matching_scores
        output_dict['ref_node_corr_knn_points']=ref_neighbor_corr_knn_points
        output_dict['src_node_corr_knn_points']=src_neighbor_corr_knn_points
        output_dict['ref_node_corr_knn_masks']=ref_neighbor_knn_masks
        output_dict['src_node_corr_knn_masks']=src_neighbor_knn_masks
        """ if self.training:

            mem_cuda=torch.cuda.memory_allocated()
            print('7',mem_cuda/(1024*1024*1024)) """
        """ loading_time = time.time() - start_time
        print("point_match",loading_time)
        start_time=time.time() """
        # 9. Generate final correspondences during testing
        if not self.training:
            if not self.with_slack:
                matching_scores = matching_scores[:, :-1, :-1]
            estimated_transforms,estimated_transforms_ori,corr_node_masks,all_ref_corr_points,all_src_corr_points,all_corr_scores= self.fine_matching(
                    ref_neighbor_corr_knn_points, src_neighbor_corr_knn_points,
                    ref_neighbor_knn_masks, src_neighbor_knn_masks,
                    matching_scores, node_corr_scores,ref_points_m,src_points_m,sym,'test'
                )
            output_dict['estimated_transforms'] = estimated_transforms
            output_dict['estimated_transforms_ori'] = estimated_transforms_ori
            output_dict['corr_node_masks'] = corr_node_masks
            output_dict['all_ref_corr_points'] = all_ref_corr_points
            output_dict['all_src_corr_points'] = all_src_corr_points
            output_dict['all_corr_scores'] = all_corr_scores
            """ loading_time = time.time() - start_time
            print("fine_matching",loading_time)
            
        if self.training:

            mem_cuda=torch.cuda.memory_allocated()
            print('8',mem_cuda/(1024*1024*1024)) """

        return output_dict


def create_model(config):
    model = MatchNet(config)
    return model


def main():
    from configs.config_scan2cad_match import config
    model = create_model(config)
    print(model)


if __name__ == '__main__':
    main()
