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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,  gt_alpha_mask,
                 image_name, uid, is_novel_view=False,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.is_novel_view = is_novel_view
        #self.depth = depth.to(torch.device(data_device))

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.original_image = None
            self.image_width = None
            self.image_height = None
        '''
        if language_feature_indices is not None:
            self.language_feature_indices = torch.from_numpy(language_feature_indices).to(self.data_device)
        else:
            self.language_feature_indices = None
        '''
        if self.original_image is not None:
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        '''查阅下要不要加
        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        '''
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_language_feature(self, language_feature_dir):#, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        #print(seg_map.shape)
        #print(feature_map.shape)
        # elif str(language_feature_name).split('.')[-1] == 'pkl':
        #     with open(language_feature_name, 'rb') as f:
        #         data = pickle.load(f)
        #     seg_map = data['seg_maps']
        #     feature_tensor = data['feature']
        # print(seg_map.shape, feature_tensor.shape)torch.Size([4, 832, 1264]) torch.Size([391, 512])
        # feature_map = torch.zeros(512, self.image_height, self.image_width)
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        #print(self.image_height, self.image_width)
        #print(x.shape)torch.Size([403004, 1])
        #print(y.shape)torch.Size([403004, 1])
        seg = seg_map[:, y, x].squeeze(-1).long()
        #print(seg.shape)torch.Size([4, 403004])
        mask = seg != -1
        #print(mask.shape)torch.Size([4, 403004])
        #print(self.image_height, self.image_width)518 778
        seg = seg.reshape(4, self.image_height, self.image_width)
        #point_feature1 = feature_map[seg[0:1]].squeeze(0)
        #mask1 = mask[0:1].reshape(1, self.image_height, self.image_width)
        #point_feature2 = feature_map[seg[1:2]].squeeze(0)
        #mask2 = mask[1:2].reshape(1, self.image_height, self.image_width)
        #point_feature3 = feature_map[seg[2:3]].squeeze(0)
        #mask3 = mask[2:3].reshape(1, self.image_height, self.image_width)
        point_feature4 = feature_map[seg[3:4]].squeeze(0)
        mask4 = mask[3:4].reshape(1, self.image_height, self.image_width)

        # point_feature = torch.cat((point_feature2, point_feature3, point_feature4), dim=-1).to('cuda')
        #point_feature1 = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        #point_feature2 = point_feature2.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        #point_feature3 = point_feature3.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        point_feature4 = point_feature4.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)

        #return point_feature1.cuda(), mask1.cuda(), point_feature2.cuda(), mask2.cuda(), \
        #       point_feature3.cuda(), mask3.cuda(), point_feature4.cuda(), mask4.cuda(), \
        #       seg[0].unsqueeze(0), seg[1].unsqueeze(0), seg[2].unsqueeze(0), seg[3].unsqueeze(0)
        return point_feature4.cuda(), mask4.cuda(), seg[3].unsqueeze(0)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

