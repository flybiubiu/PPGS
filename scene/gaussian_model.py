#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import random
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.lem_utils import Embedding
from scene.index_decoder import IndexDecoder
from scene.fushion_features_net import *

class GaussianModel:
#用于设置一些激活函数和变换函数
    def setup_functions(self):
        # 构建协方差矩阵，该函数接受 scaling（尺度）、scaling_modifier（尺度修正因子）、rotation（旋转）作为参数
        # 与原文一致
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        # 将不透明度激活函数设置为 sigmoid 函数，保证（0，1）
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        ## 用于归一化旋转矩阵
        self.rotation_activation = torch.nn.functional.normalize
        self.positional_encoding = Embedding(3, 4)


    def __init__(self, sh_degree : int, semantic_features_dim: int, points_num_limit: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.semantic_features_dim = semantic_features_dim
        self.points_num_limit = points_num_limit  # to limit the number of points in the model
        #空间位置
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._semantic_features = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._uncertainty = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._semantic_features,
            self._scaling,
            self._rotation,
            self._opacity,
            self._uncertainty,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._semantic_features,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._uncertainty,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_xyz_feature(self):
        return self.positional_encoding(self._xyz)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        #print(features_dc.shape, features_rest.shape)
        #torch.Size([112627, 1, 3]) torch.Size([112627, 15, 3]) [25/10 00:37:44]
        return torch.cat((features_dc, features_rest), dim=1)
        #return features_dc#, features_rest

    @property
    def get_semantic_features(self):
        return self._semantic_features
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    @property
    def get_uncertainty(self):
        return self.opacity_activation(self._uncertainty)


    def set_color(self, mask, rgb):
        self._features_dc.requires_grad = False
        self._features_dc[:, :, :][mask] = RGB2SH(torch.tensor(rgb)).to(self._features_dc.device)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # 所有相机的中心点位置到最远camera的距离 * 1.1
        # 根据大量的3DGS解读，应该与学习率有关，防止固定的学习率适配不同尺度的场景时出现问题。
        self.spatial_lr_scale = spatial_lr_scale
        #print(spatial_lr_scale)4.960308933258057 garden 数据集
        # 点云转tensor送入GPU，实际上就是稀疏点云的3D坐标
        # (N, 3) 这里输出的是所有点云
        #print(pcd.points.shape)(138766, 3)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        #print(features.shape)torch.Size([112627, 3, 16]) [25/10 16:10:03]
        #print(features[:, 3:, 1:].shape)torch.Size([112627, 0, 15]) [25/10 16:11:19]
        #print(features[:, :3, 0 ].shape)torch.Size([112627, 3])
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        semantic_features = torch.zeros((fused_color.shape[0], self.semantic_features_dim)).float().cuda()
        #print(semantic_features.shape)torch.Size([54275, 8])
        nn.init.normal_(semantic_features, mean=0.0, std=1.0)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 调用simple_knn的distCUDA2函数，计算点云中的每个点到与其最近的K个点的平均距离的平方
        # dist2的大小应该是(N,)。
        # 首先可以明确的是这句话用来初始化scale，且scale（的平方）不能低于1e-7。
        # 我阅读了一下submodules/simple-knn/simple_knn.cu，大致猜出来了这个是什么意思。
        # distCUDA2函数由simple_knn.cu的SimpleKNN::knn函数实现。
        # KNN意思是K-Nearest Neighbor，即求每一点最近的K个点。
        # simple_knn.cu中令k=3，求得每一点最近的三个点距该点的平均距离。
        # 原理是把3D空间中的每个点用莫顿编码（Morton Encoding）转化为一个1D坐标
        # 用到了能够填满空间的Z曲线
        # 然后对1D坐标进行排序，从而确定离每个点最近的三个点。
        # simple_knn.cu实际上还用了一种加速策略，是将点集分为多个大小为1024的块（box），
        # 在每个块内确定3个最近邻居和它们的平均距离。用平均距离作为Gaussian的scale。
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        # 因为2DGS中只有2个缩放值。
        # 因为scale的激活函数是exp，所以这里存的也不是真的scale，而是ln(scale)。
        # 注意dist2其实是距离的平方，所以这里要开根号。
        # repeat(1, 2)标明两个方向上scale的初始值是相等的。
        # scales的大小：(N, 2) 这是与3DGS完全不同的。
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        # 2DGS不是，2DGS使用[0,1]的均匀分布进行初始化
        # 这里与3DGS有明显区别
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        # 完全不同，这里使用：torch.log(x/(1-x))，而不是sigmoid。
        # 因为输入时，透明度是（N, 1）,这里统一后的初始值为-2.1972
        # 原因不明,但这里最终的值，与3DGS保持一致（-2.197）
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        uncertainty = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._semantic_features = nn.Parameter(semantic_features.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._uncertainty = nn.Parameter(uncertainty.requires_grad_(True))
        # 投影到2D时, 每个2D gaussian最大的半径，这里初始为（N，）
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._semantic_features], 'lr': training_args.index_lr, "name": "clip_f_indices"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._uncertainty], 'lr': training_args.opacity_lr, "name": "uncertainty"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._semantic_features.shape[1]):
            l.append('clip_f_indices_{}'.format(i))
        l.append('opacity')
        l.append('uncertainty')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        clip_f_indices = self._semantic_features.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        uncertainty = self._uncertainty.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, clip_f_indices, opacities, uncertainty, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def save_lem_ply(self, path, index_decoder: IndexDecoder, color_map):
        mkdir_p(os.path.dirname(path))

        with torch.no_grad():
            #decoded_clip_feat_indices = index_decoder(self._semantic_features[..., None, None]).squeeze()  # (N, C=128)
            decoded_clip_feat_indices = self._semantic_features[..., None, None].squeeze()  # (N, C=128)
            # color_maps: (128, 3)
            temp = 0.5  # ->0 = argmax, ->+inf = unifrom
            prob_tensor = torch.softmax(decoded_clip_feat_indices / temp, dim=1)  # (N, C=128)
            feat_indices_rgb = torch.einsum('nc,ck->nk', prob_tensor, color_map.to(prob_tensor.device))  # (N, 3)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        clip_f_indices = self._semantic_features.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        uncertainty = self._uncertainty.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        feat_indices_rgb = feat_indices_rgb.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        for i in range(3):
            dtype_full.append((f"lemcolor_{i}", 'f4'))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, clip_f_indices, opacities, scale, uncertainty, rotation, feat_indices_rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        uncertainty = np.asarray(plydata.elements[0]["uncertainty"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        clip_f_indices_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("clip_f_indices_")]
        clip_f_indices_names = sorted(clip_f_indices_names, key=lambda x: int(x.split('_')[-1]))
        clip_f_indices = np.zeros((xyz.shape[0], len(clip_f_indices_names)))
        for idx, attr_name in enumerate(clip_f_indices_names):
            clip_f_indices[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._semantic_features = nn.Parameter(torch.tensor(clip_f_indices, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._uncertainty = nn.Parameter(torch.tensor(uncertainty, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    '''
    def forward(self, viewpoint_camera, store_cache=False):
        fushion_model = FushionModel(dataset.fusion_features_dim, dataset.semantic_features_dim, view_dim=None,multires=[10, 0]).to("cuda")
        fushion_optim = torch.optim.AdamW(fushion_model.parameters(), lr=opt.fushion_model_lr, weight_decay=1e-5)
        fushion_model.train()
        self._semantic_features = self.fusion_net(viewpoint_camera, self._semantic_features)
    '''

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._semantic_features = optimizable_tensors["clip_f_indices"]
        self._opacity = optimizable_tensors["opacity"]
        self._uncertainty = optimizable_tensors["uncertainty"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_semantic_features, new_opacities, new_uncertainty, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "clip_f_indices": new_semantic_features,
        "opacity": new_opacities,
        "uncertainty": new_uncertainty,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._semantic_features = optimizable_tensors["clip_f_indices"]
        self._opacity = optimizable_tensors["opacity"]
        self._uncertainty = optimizable_tensors["uncertainty"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        #print(N)2
        # 获取初始点的数量。
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充到其中。
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 创建一个掩码，标记那些梯度大于等于指定阈值的点。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 一步过滤掉那些缩放（scaling）大于一定百分比的场景范围的点
        # 这里是一个高斯分裂的过程：被分裂的Gaussians满足两个条件：
        # 		1. （平均）梯度过大；
        # 		2. 在某个方向的最大缩放大于一个阈值。
        # 		参照论文5.2节“On the other hand...”一段，大Gaussian被分裂成两个小Gaussians，
        # 		其放缩被除以φ=1.6，且位置是以原先的大Gaussian作为概率密度函数进行采样的。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        surplus = self.points_num_limit - n_init_points
        if surplus <= 0:
            return
        current_true_count = selected_pts_mask.sum().item()
        # If current_true_count > surplus, we need randomly select surplus(num) points to clone/split
        if current_true_count > surplus:
            true_indices = torch.where(selected_pts_mask)[0]
            random_indices = torch.randperm(current_true_count)[:surplus]
            new_selected_pts_mask = torch.zeros_like(selected_pts_mask)
            new_selected_pts_mask[true_indices[random_indices]] = True
            selected_pts_mask = new_selected_pts_mask

            # 为每个点生成新的样本，其中 stds 是点的缩放，means 是均值， 第一步是一样的
        # 这里从新的点云中更新缩放因子，并且进行同样的复制一份
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # 这里我大致明白，本身获取的是（su, sv）,为了与旋转矩阵相对应，构建（su,sv,0）
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        # 这里就是一个同样大小的矩阵
        means = torch.zeros_like(stds)
        # 使用均值和标准差生成样本
        samples = torch.normal(mean=means, std=stds)
        # 为每个点构建旋转矩阵，并将其重复 N 次。
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 将旋转后的样本点添加到原始点的位置。
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 生成新的缩放参数
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 将旋转、原始点特征、等等重复N次
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_semantic_features = self._semantic_features[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_uncertainty = self._uncertainty[selected_pts_mask].repeat(N, 1)

        # 调用另一个方法 densification_postfix，该方法对新生成的点执行后处理操作（此处跟densify_and_clone一样）。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_semantic_features, new_opacity, new_uncertainty, new_scaling, new_rotation)
        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后。
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 根据修剪过滤器，修剪模型中的一些参数。
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # 建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False。
        # 提取出大于阈值`grad_threshold`且缩放参数较小（小于self.percent_dense * scene_extent）的Gaussians，在下面进行克隆
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）的场景范围（scene_extent）的点。
        # 这样可以确保新添加的点不会太远离原始数据。
        # 根据掩码选取符合条件的点的其他特征，如颜色、透明度、缩放和旋转等。

        n_init_points = self.get_xyz.shape[0]
        surplus = self.points_num_limit - n_init_points
        if surplus <= 0:
            return
        current_true_count = selected_pts_mask.sum().item()
        # If current_true_count > surplus, we need randomly select surplus(num) points to clone/split
        if current_true_count > surplus:
            true_indices = torch.where(selected_pts_mask)[0]
            random_indices = torch.randperm(current_true_count)[:surplus]
            new_selected_pts_mask = torch.zeros_like(selected_pts_mask)
            new_selected_pts_mask[true_indices[random_indices]] = True
            selected_pts_mask = new_selected_pts_mask

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_semantic_features = self._semantic_features[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_uncertainty = self._uncertainty[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_semantic_features, new_opacities, new_uncertainty, new_scaling, new_rotation)

    # 执行密集化和修剪操作
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 计算密度估计的梯度
        grads = self.xyz_gradient_accum / self.denom
        # 将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 接下来移除一些Gaussians，它们满足下列要求中的一个：
        # 1. 接近透明（不透明度小于min_opacity）
        # 2. 在某个相机视野里出现过的最大2D半径大于屏幕（像平面）大小
        # 3. 在某个方向的最大缩放大于0.1 * extent（也就是说很长的长条形也是会被移除的）
        # 创建一个掩码，标记那些透明度小于指定阈值的点。.squeeze() 用于去除掩码中的单维度。
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # 设置相机的范围
        if max_screen_size:
            # 创建一个掩码，标记在图像空间中半径大于指定阈值的点。
            big_points_vs = self.max_radii2D > max_screen_size
            # 创建一个掩码，标记在世界空间中尺寸大于指定阈值的点。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 将这两个掩码与先前的透明度掩码进行逻辑或操作，得到最终的修剪掩码。
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 根据修剪掩码，修剪模型中的一些参数
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

# 统计坐标的累积梯度和均值的分母（即迭代步数）
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def scene_params_zero_grad(self):
        self._xyz.grad.zero_()
        self._features_dc.grad.zero_()
        self._features_rest.grad.zero_()
        self._opacity.grad.zero_()
        self._scaling.grad.zero_()
        self._rotation.grad.zero_()