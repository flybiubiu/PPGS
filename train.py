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

import os
import torch
import yaml
from random import randint
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.index_decoder import *
from scene.fushion_features_net import *
from scene.dynamic_loss import DynamicWeightedLoss
from utils.general_utils import safe_state
from utils.lem_utils import *
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
import configargparse
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib
from utils.sh_utils import SH2RGB

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
print("TENSORBOARD_FOUND is: ", TENSORBOARD_FOUND)
FIRST_REPORT = True

'''
cmapper = matplotlib.colormaps.get_cmap('jet_r')
def colorize_with_mask(depthlist, background=(0, 0, 0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    #print(depthlist.shape)(1, 1, 518, 778)
    batch, vx, vy = np.where(depthlist != 0)
    if dmindmax is None:
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax

    norm_dth = np.ones_like(depthlist) * dmax  # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy] - dmin) / (dmax - dmin)

    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1, 1, 1, 3)  # [B, H, W, 3]
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch, vx, vy, :3]

    return final_depth
'''
'''
def compute_semantic_loss(language_feature_indices, gt_language_feature_indices, uncertainty, ce):
    # Lem Loss
    # Resize clip feature indices to match image size
    #print(gt_language_feature_indices.shape)torch.Size([1, 60, 90])
    a, b = language_feature_indices.shape[1], language_feature_indices.shape[2]#822, 1236

    upsampled_gt_language_feature_indices = \
        F.interpolate(gt_language_feature_indices.unsqueeze(0).float(), size=(a, b), mode='nearest').squeeze(0).long()
    #print(upsampled_gt_language_feature_indices.shape)torch.Size([1, 822, 1236])

    upsampled_gt_language_feature_indices = upsampled_gt_language_feature_indices.permute(1, 2, 0).view(-1)
    #print(upsampled_gt_language_feature_indices.shape)torch.Size([1015992])

    language_feature_indices = language_feature_indices.permute(1, 2, 0)
    language_feature_indices = language_feature_indices.reshape(-1, language_feature_indices.shape[-1])
    #print(language_feature_indices.shape)torch.Size([1015992, 128])
    semantic_loss = ce(language_feature_indices, upsampled_gt_language_feature_indices)
    #print(semantic_loss.shape)torch.Size([1015992])

    uncertainty = 1.0 - uncertainty.permute(1, 2, 0).reshape(-1)
    semantic_loss = (semantic_loss * uncertainty).mean()

    return semantic_loss
'''

def compute_semantic_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def compute_mlp_smooth_loss(xyzs, embedding, decoder, gs_semantic_features, rgb_features, uncertainty, smooth_loss_uncertainty_min):
    #print(xyzs.shape)torch.Size([54275, 3])
    xyzs_pe = embedding(xyzs)
    #print(xyzs_pe.shape)
    #sys.exit()
    #print(xyzs_pe.shape)torch.Size([54275, 3])
    #xyzs_features = decoder(xyzs_pe)
    xyzs_features = decoder(xyzs_pe, alpha = None)
    xyzs_features_alpha = decoder(xyzs_pe, alpha = True)
    #print(xyzs_features.shape)torch.Size([54275, 8])
    sem_mlp_loss = ((xyzs_features - gs_semantic_features.detach()) ** 2).mean(dim=1)
    #print(xyz_mlp_loss.shape)torch.Size([54275])
    rgb_mlp_loss = ((xyzs_features_alpha - rgb_features.detach()) ** 2).mean(dim=1)

    weights = (1 - uncertainty) * smooth_loss_uncertainty_min + uncertainty * 1.0
    #print(weights.shape)torch.Size([54275])
    rgb_features_smooth_loss = ((xyzs_features_alpha.detach() - rgb_features) ** 2).mean(dim=1) * weights
    semantic_features_smooth_loss = ((xyzs_features.detach() - gs_semantic_features) ** 2).mean(dim=1) * weights
    #print(semantic_features_smooth_loss.shape)torch.Size([54275])
    return sem_mlp_loss.mean(), rgb_mlp_loss.mean(), rgb_features_smooth_loss.mean(), semantic_features_smooth_loss.mean()
    #return 0, rgb_mlp_loss.mean(), rgb_features_smooth_loss.mean(), 0

def training(dataset, opt, pipe,
             testing_iterations, test_set,
             saving_iterations, checkpoint_iterations,
             checkpoint, debug_from):
    #print(testing_iterations)
    #sys.exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, dataset.semantic_features_dim, dataset.points_num_limit)
    scene = Scene(dataset, gaussians, test_set=test_set)#创建一个 Scene 类的实例，使用数据集和之前创建的 GaussianModel 实例作为参数。
    # 这里非常重要，此时已经初始化了整个高斯的点云。
    gaussians.training_setup(opt)

    #index_decoder = IndexDecoder(dataset.semantic_features_dim, dataset.codebook_size).to("cuda")
    #decoder_optim = torch.optim.AdamW(index_decoder.parameters(), lr=opt.decoder_lr, weight_decay=1e-5)
    #ce = torch.nn.CrossEntropyLoss(reduction='none')
    #index_decoder.train()#8维feature到codebook的size 128维

    #fushion_model = FushionModel(dataset.fusion_features_dim, dataset.semantic_features_dim, view_dim = None, multires = [10, 0]).to("cuda")
    #fushion_optim = torch.optim.AdamW(fushion_model.parameters(), lr=opt.fushion_model_lr, weight_decay=1e-5)
    #fushion_model.train()

    dataset.xyz_encoding_in_channels_xyz = 3 * (2 * dataset.xyz_embedding_N_freqs + 1)
    #print(dataset.xyz_embedding_N_freqs)0
    #print(dataset.xyz_encoding_in_channels_xyz)
    #sys.exit()
    xyz_embedding = Embedding(3, dataset.xyz_embedding_N_freqs).to("cuda")
    xyz_decoder = XyzMLP(D=dataset.xyz_encoding_D,
                         W=dataset.xyz_encoding_W,
                         in_channels_xyz=dataset.xyz_encoding_in_channels_xyz,
                         out_channels_xyz=dataset.xyz_encoding_out_channels_xyz).to("cuda")
    xyz_decoder_optim = torch.optim.AdamW(xyz_decoder.parameters(), lr=opt.decoder_lr, weight_decay=1e-5)
    xyz_decoder.train()#xyz坐标3维变成 8维

    dynamic_loss_weight = DynamicWeightedLoss().to("cuda")
    dynamic_loss_weight_optim = torch.optim.Adam(dynamic_loss_weight.parameters(), lr=opt.dynamic_lr, weight_decay=1e-5)
    dynamic_loss_weight.train()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        #index_decoder_ckpt = os.path.join(os.path.dirname(checkpoint), "index_decoder_" + os.path.basename(checkpoint))
        #index_decoder.load_state_dict(torch.load(index_decoder_ckpt))

    bg_color = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_semantics_loss_for_log = 0.0
    #ema_semantics1_loss_for_log = 0.0
    #ema_semantics2_loss_for_log = 0.0
    #ema_semantics3_loss_for_log = 0.0
    ema_semantics4_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    color_map = generate_colors(8)#dataset.codebook_size)

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        #print(viewpoint_cam.image_name)
        #print("******")

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        semantic_features = render_pkg["semantic_features"]
        #print(semantic_features.shape)torch.Size([8, 822, 1236])
        uncertainty = render_pkg["uncertainty"]
        #print(uncertainty.shape)torch.Size([1, 822, 1236])

        # Index Features Regularization Loss
        #xyzs = gaussians.get_xyz.detach()
        #gs_semantic_features = gaussians.get_semantic_features
        #alpha_features = gaussians.get_opacity
        #sh_features = gaussians.get_features[:,:1 ,:].squeeze(dim=1)
        #rgb_features = SH2RGB(sh_features)
        #print(features_rest.shape)
        #print(rgb_features.shape)torch.Size([112627, 16, 3]) [25/10 00:02:25]
        
        #print(gs_semantic_features.shape)torch.Size([54275, 8])
        #####
        #gaussians._semantic_features = fushion_model(gaussians._semantic_features, xyzs)
        '''
        i = len(xyzs) // 2
        gs_semantic_features1 = fushion_model(gs_semantic_features[:i], xyzs[:i])
        torch.cuda.empty_cache()
        gs_semantic_features2 = fushion_model(gs_semantic_features[i:], xyzs[i:])
        torch.cuda.empty_cache()
        gs_semantic_features = torch.cat((gs_semantic_features1, gs_semantic_features2), 0)
        '''
        #gs_semantic_features = fushion_model(gs_semantic_features, xyzs)
        #####
        #gs_uncertainty = gaussians.get_uncertainty
        #print(gs_uncertainty.shape)torch.Size([54275, 1])
        #print(dataset.smooth_loss_uncertainty_min)0.1

        if iteration % 10 == 0:
            sem_mlp_loss, rgb_mlp_loss, rgb_features_smooth_loss,semantic_features_smooth_loss = compute_mlp_smooth_loss(gaussians.get_xyz.detach(), xyz_embedding, xyz_decoder, gaussians.get_semantic_features, SH2RGB(gaussians.get_features[:,:1 ,:].squeeze(dim=1)),
                                                                gaussians.get_uncertainty.squeeze().detach(),
                                                                dataset.smooth_loss_uncertainty_min)
        else:
            sem_mlp_loss, rgb_mlp_loss, rgb_features_smooth_loss,semantic_features_smooth_loss = 0.0, 0.0, 0.0, 0.0

        # Lem Loss
        norm_semantic_features = F.normalize(semantic_features, p=2, dim=0)
        #print(semantic_features.shape)torch.Size([8, 822, 1236])
        #print(norm_semantic_features.shape)torch.Size([8, 822, 1236])
        language_feature_indices = norm_semantic_features#index_decoder(norm_semantic_features.unsqueeze(0)).squeeze(0)
        #print(language_feature_indices.shape)torch.Size([128, 822, 1236])
        #gt_language_feature_indices = viewpoint_cam.language_feature_indices.permute(2, 0, 1)
        #gt_language_feature_indices1, language_feature_mask1, gt_language_feature_indices2, language_feature_mask2, gt_language_feature_indices3, language_feature_mask3, gt_language_feature_indices4, language_feature_mask4, seg1, seg2, seg3, seg4\
        gt_language_feature_indices4, language_feature_mask4, seg4 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path)#, feature_level=dataset.feature_level)
        #print(gt_language_feature_indices.shape)torch.Size([1, 60, 90])
        #print(uncertainty.shape)torch.Size([1, 822, 1236]) [10/08 18:58:16]
        #semantic_loss = compute_semantic_loss(language_feature_indices, gt_language_feature_indices, uncertainty, ce)
        #gt_language_feature_indices1 = F.normalize(gt_language_feature_indices1, p=2, dim=0)
        #semantic_loss1 = compute_semantic_loss(language_feature_indices*language_feature_mask1,
        #                                       gt_language_feature_indices1*language_feature_mask1)
        #semantic_loss2 = compute_semantic_loss(language_feature_indices * language_feature_mask2,
        #                                       gt_language_feature_indices2 * language_feature_mask2)
        #semantic_loss3 = compute_semantic_loss(language_feature_indices * language_feature_mask3,
        #                                       gt_language_feature_indices3 * language_feature_mask3)
        semantic_loss4 = compute_semantic_loss(language_feature_indices * language_feature_mask4,
                                               gt_language_feature_indices4 * language_feature_mask4)
        #semantic_loss = semantic_loss1 + semantic_loss2 + semantic_loss3 + semantic_loss4
        uncertainty_loss = torch.mean(-torch.log(1 - uncertainty))

        depth = render_pkg["surf_depth"]
        norm = depth.max()
        depth = depth / norm
        #gt_depth = viewpoint_cam.depth

        # Recon Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        '''
        # loss
        #total_loss = loss + dist_loss + normal_loss
        semantic_loss = dataset.semantic_loss_weight1 * semantic_loss1 \
                     + dataset.semantic_loss_weight2 * semantic_loss2 \
                     + dataset.semantic_loss_weight3 * semantic_loss3 \
                     + dataset.semantic_loss_weight4 * semantic_loss4
        '''
        semantic_loss = dataset.semantic_loss_weight4 * semantic_loss4 + sem_mlp_loss + semantic_features_smooth_loss
        res_loss = (dataset.reconstruction_loss_weight * loss \
                     + dataset.uncertainty_loss_weight * uncertainty_loss \
                     + dataset.xyzmlp_loss_weight * rgb_mlp_loss \
                     + dataset.smooth_loss_weight * rgb_features_smooth_loss \
                     + dist_loss + normal_loss)
        #print(res_loss, semantic_loss)
        loss_dict = {"rgb": res_loss, "semantics": semantic_loss}
        total_loss = dynamic_loss_weight(loss_dict)
        #print(total_loss)

        #total_loss = res_loss + semantic_loss
        #print(total_loss)

        
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            #ema_semantics1_loss_for_log = 0.4 * semantic_loss1.item() + 0.6 * ema_semantics1_loss_for_log
            #ema_semantics2_loss_for_log = 0.4 * semantic_loss2.item() + 0.6 * ema_semantics2_loss_for_log
            #ema_semantics3_loss_for_log = 0.4 * semantic_loss3.item() + 0.6 * ema_semantics3_loss_for_log
            ema_semantics4_loss_for_log = 0.4 * semantic_loss4.item() + 0.6 * ema_semantics4_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            ema_semantics_loss_for_log = 0.4 * semantic_loss.item() + 0.6 * ema_semantics_loss_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Semantics_Loss": f"{ema_semantics_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss_for_log', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss_for_log', ema_normal_for_log, iteration)
                #tb_writer.add_scalar('train_loss_patches/semantics1_Loss_for_log', ema_semantics1_loss_for_log, iteration)
                #tb_writer.add_scalar('train_loss_patches/semantics2_Loss_for_log', ema_semantics2_loss_for_log, iteration)
                #tb_writer.add_scalar('train_loss_patches/semantics3_Loss_for_log', ema_semantics3_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/semantics4_Loss_for_log', ema_semantics4_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/semanticstotal_Loss_for_log', ema_semantics_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/res_loss', res_loss, iteration)
                tb_writer.add_scalar('train_loss_patches/semantic_loss', semantic_loss, iteration)
                tb_writer.add_scalar('train_loss_patches/total_loss', total_loss, iteration)


            training_report(tb_writer, iteration, viewpoint_cam.is_novel_view,
                            color_map, dataset,
                            Ll1, loss, semantic_loss, l1_loss, 0, 0, uncertainty_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            scene,  render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                #index_decoder.eval()
                scene.save(iteration, None, color_map)
                #index_decoder.train()


            # Densification
            # 在一定的迭代次数内进行密集化处理
            if iteration < opt.densify_until_iter:
                # 将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                #白色背景
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                #decoder_optim.step()
                #decoder_optim.zero_grad(set_to_none=True)
                #fushion_optim.step()
                #fushion_optim.zero_grad(set_to_none = True)
                xyz_decoder_optim.step()
                xyz_decoder_optim.zero_grad(set_to_none=True)
                dynamic_loss_weight_optim.step()
                dynamic_loss_weight_optim.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                #torch.save(index_decoder.state_dict(), scene.model_path + "/index_decoder_chkpnt" + str(iteration) + ".pth")
                torch.save(xyz_decoder.state_dict(), scene.model_path + "/xyz_decoder_chkpnt" + str(iteration) + ".pth")
                #torch.save(fushion_model.state_dict(), scene.model_path + "/fushion_model_chkpnt" + str(iteration) + ".pth")
                torch.save(dynamic_loss_weight.state_dict(), scene.model_path + "/dynamic_loss_weight_chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(dataset, opt, pipe):
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))
    # Save all the params
    with open(os.path.join(dataset.model_path, "cfg_args.yml"), 'w') as cfg_log_f:
        args = {
            "ModelParams" : vars(dataset),
            "PipelineParams" : vars(pipe),
            "OptimizationParams" : vars(opt),
        }
        yaml.dump(args, cfg_log_f)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, is_novel_view,
                    color_maps, dataset,
                    Ll1, loss, semantic_loss, l1_loss, xyz_mlp_loss, smooth_loss, uncertainty_loss,
                    elapsed, testing_iterations,
                    scene : Scene, renderFunc, renderArgs):
    #gt_language_feature_indices = gt_language_feature_indices.type(torch.int64)
    #language_feature_mask = language_feature_mask.type(torch.int64)
    global FIRST_REPORT
    if tb_writer:
        if not is_novel_view:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        #tb_writer.add_scalar('train_loss_patches/semantic_loss', semantic_loss.item(), iteration)
        if smooth_loss > 0.0:
            tb_writer.add_scalar('train_loss_patches/smooth_loss', smooth_loss.item(), iteration)
        if xyz_mlp_loss > 0.0:
            tb_writer.add_scalar('train_loss_patches/xyz_mlp_loss', xyz_mlp_loss.item(), iteration)
        if uncertainty_loss > 0.0:
            tb_writer.add_scalar('train_loss_patches/uncertainty_loss', uncertainty_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    #print(testing_iterations)
    #sys.exit()
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                semantic_loss_test = 0.0
                uncertainty_test = 0.0
                # mkdir for test rendering
                if config['name'] == "test":
                    if FIRST_REPORT:
                        os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_gt_images"), exist_ok=True)
                        os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_gt_indices"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_images"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_indices"), exist_ok=True)

                for idx, viewpoint in enumerate(config['cameras']):
                    #print(idx, viewpoint.image_name)
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    #gt_language_feature_indices1, language_feature_mask1, gt_language_feature_indices2, language_feature_mask2, gt_language_feature_indices3, language_feature_mask3, gt_language_feature_indices4, language_feature_mask4, seg1, seg2, seg3, seg4\
                    gt_language_feature_indices4, language_feature_mask4, seg4    = viewpoint.get_language_feature(language_feature_dir=dataset.lf_path)#, feature_level=dataset.feature_level)
                    #gt_language_feature_indices1 = F.normalize(gt_language_feature_indices1, p=2, dim=0)
                    #gt_language_feature_indices2 = F.normalize(gt_language_feature_indices2, p=2, dim=0)
                    #gt_language_feature_indices3 = F.normalize(gt_language_feature_indices3, p=2, dim=0)
                    gt_language_feature_indices4 = F.normalize(gt_language_feature_indices4, p=2, dim=0)

                    language_feature_indices = F.normalize(render_result["semantic_features"], p=2, dim=0)
                    pca_feat_image = pca(language_feature_indices)  # already in [0.0, 1.0]
                    #gt_language_feature_indices = F.normalize(gt_language_feature_indices, p=2, dim=0)
                    #gt_pca_feat_image1 = pca(gt_language_feature_indices1)  # already in [0.0, 1.0]
                    #gt_pca_feat_image2 = pca(gt_language_feature_indices2)
                    #gt_pca_feat_image3 = pca(gt_language_feature_indices3)
                    gt_pca_feat_image4 = pca(gt_language_feature_indices4)
                    decoded_clip_feat_indices = language_feature_indices.unsqueeze(0)#F.normalize(render_result["semantic_features"], p=2, dim=0).unsqueeze(0)#)

                    # color_maps: (128, 3)
                    temp = 1  # ->0 = argmax, ->+inf = unifrom
                    prob_tensor_1 = torch.softmax(decoded_clip_feat_indices / temp, dim=1)  # (N, C=128, H, W)
                    feat_indices_image_1 = torch.einsum('nchw,ck->nkhw', prob_tensor_1,
                                                        color_maps.to(prob_tensor_1.device))  # (N, 3, H, W)

                    temp = 0.01  # ->0 = argmax, ->+inf = unifrom
                    prob_tensor_001 = torch.softmax(decoded_clip_feat_indices / temp, dim=1)  # (N, C=128, H, W)
                    feat_indices_image_001 = torch.einsum('nchw,ck->nkhw', prob_tensor_001,
                                                          color_maps.to(prob_tensor_001.device))  # (N, 3, H, W)

                    uncertainty = torch.clamp(render_result["uncertainty"], 0.0, 1.0)

                    if tb_writer and (idx < 50):
                        from utils.general_utils import colormap
                        depth = render_result["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        #print(depth.shape)torch.Size([1, 518, 778])
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render_uncertainty".format(viewpoint.image_name),
                            uncertainty[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render_indices_softmax".format(viewpoint.image_name),
                            feat_indices_image_1, global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_indices".format(viewpoint.image_name),
                                             feat_indices_image_001, global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render_indices_feat_pca3".format(viewpoint.image_name),
                            pca_feat_image[None], global_step=iteration)
                        '''
                        tb_writer.add_images(
                            config['name'] + "_view_{}/gt_render_indices_feat1_pca3".format(viewpoint.image_name),
                            gt_pca_feat_image1[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + "_view_{}/gt_render_indices_feat2_pca3".format(viewpoint.image_name),
                            gt_pca_feat_image2[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + "_view_{}/gt_render_indices_feat3_pca3".format(viewpoint.image_name),
                            gt_pca_feat_image3[None], global_step=iteration)
                        '''
                        tb_writer.add_images(
                            config['name'] + "_view_{}/gt_render_indices_feat4_pca3".format(viewpoint.image_name),
                            gt_pca_feat_image4[None], global_step=iteration)
                        if FIRST_REPORT:
                            #print(gt_language_feature_indices.shape)torch.Size([8, 518, 778]) [02/10 10:25:47]
                            #print(gt_language_feature_indices.dtype)
                            #print(language_feature_mask.dtype)
                            #print(gt_language_feature_indices.device)
                            #print(language_feature_mask.device)
                            '''
                            gt_indices = F.embedding(gt_language_feature_indices, color_maps.to(
                                viewpoint.language_feature_indices.device)).squeeze()#.permute(2, 0, 1)
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth_indices".format(viewpoint.image_name),
                                gt_indices[None], global_step=iteration)
                            '''
                            #print(language_feature_mask.shape)torch.Size([1, 518, 778])
                            #print(language_feature_mask.permute(2, 0, 1).shape)torch.Size([778, 1, 518])
                            #print(color_maps.shape)torch.Size([8, 3])
                            #gt_mask = F.embedding(language_feature_mask.permute(1, 2, 0), color_maps.to(
                                #viewpoint.language_feature_indices.device)).squeeze().permute(2, 0, 1)
                            #print(gt_mask.shape)torch.Size([3, 518, 778])
                            #print(seg1.shape)
                            #seg1_norm = seg1.max()
                            #seg1_mask = seg1 / seg1_norm
                            '''
                            seg1 = colormap(seg1.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth_mask1".format(viewpoint.image_name),
                                seg1[None], global_step=iteration)
                            #seg2_norm = seg2.max()
                            #seg2_mask = seg2 / seg2_norm
                            seg2 = colormap(seg2.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth_mask2".format(viewpoint.image_name),
                                seg2[None], global_step=iteration)
                            #seg3_norm = seg3.max()
                            #seg3_mask = seg3 / seg3_norm
                            seg3 = colormap(seg3.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth_mask3".format(viewpoint.image_name),
                                seg3[None], global_step=iteration)
                            #seg4_norm = seg4.max()
                            #seg4_mask = seg4 / seg4_norm
                            '''
                            seg4 = colormap(seg4.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth_mask4".format(viewpoint.image_name),
                                seg4[None], global_step=iteration)
                            '''
                            upsampled_gt_indices = F.interpolate(gt_indices.unsqueeze(0),
                                                                 size=(gt_image.shape[1], gt_image.shape[2]),
                                                                 mode='nearest')
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth_upsampled_indices".format(viewpoint.image_name),
                                upsampled_gt_indices, global_step=iteration)
                            '''
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                        try:
                            rend_alpha = render_result['rend_alpha']
                            rend_normal = render_result["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_result["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_result["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    '''
                    semantic_loss_test1 = compute_semantic_loss(
                        language_feature_indices*language_feature_mask1,
                        gt_language_feature_indices1*language_feature_mask1
                        #render_result["uncertainty"],
                        #torch.nn.CrossEntropyLoss(reduction='none')
                        ).double()
                    semantic_loss_test2 = compute_semantic_loss(
                        language_feature_indices * language_feature_mask2,
                        gt_language_feature_indices2 * language_feature_mask2
                        # render_result["uncertainty"],
                        # torch.nn.CrossEntropyLoss(reduction='none')
                    ).double()
                    semantic_loss_test3 = compute_semantic_loss(
                        language_feature_indices * language_feature_mask3,
                        gt_language_feature_indices3 * language_feature_mask3
                        # render_result["uncertainty"],
                        # torch.nn.CrossEntropyLoss(reduction='none')
                    ).double()
                    '''
                    semantic_loss_test4 = compute_semantic_loss(
                        language_feature_indices * language_feature_mask4,
                        gt_language_feature_indices4 * language_feature_mask4
                        # render_result["uncertainty"],
                        # torch.nn.CrossEntropyLoss(reduction='none')
                    ).double()
                    semantic_loss_test = semantic_loss_test4#(semantic_loss_test1 + semantic_loss_test2 +
                                          #semantic_loss_test3 + semantic_loss_test4)
                    uncertainty_test += torch.mean(render_result["uncertainty"]).double()
                    torch.cuda.empty_cache()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                semantic_loss_test /= len(config['cameras'])
                uncertainty_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} semantic_loss {} uncert {}"
                      .format(iteration, config['name'], l1_test, psnr_test, semantic_loss_test, uncertainty_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - semantic_loss', semantic_loss_test,
                                         iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - uncertainty', uncertainty_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        if FIRST_REPORT:
            FIRST_REPORT = False

if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgParser(description="Training script parameters")
    lp = ModelParams(parser)
    ## 本身模型属性可修改内容：sh_degree，source_path，model_path，image属性，
    # resolution，white_background，data_device，eval，render_items
    op = OptimizationParams(parser)
    #convert_SHs_python, compute_cov3D_python, depth_ratio, debug四个参数
    pp = PipelineParams(parser)
    parser.add('--ip', type=str, default="127.0.0.1")
    parser.add('--exper_name', type=str, default="")
    parser.add('--config', required=True, is_config_file=True, help='config file path')
    parser.add('--debug_from', type=int, default=-1)
    parser.add('--port', type=int, default=6006)
    parser.add('--detect_anomaly', action='store_true', default=False)#检测梯度异常
    parser.add("--test_iterations", nargs="+", type=int, default=[0]+[i for i in range(0, 30_001, 5000)])
    parser.add("--test_set", nargs="+", type=str, default=[])
    parser.add("--save_iterations", nargs="+", type=int, default=[i for i in range(0, 30_001, 5000)])
    parser.add("--quiet", action="store_true")
    parser.add("--checkpoint_iterations", nargs="+", type=int, default=[i for i in range(0, 30_001, 5000)])
    parser.add("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args.exper_name)
    print(args.model_path)
    args.model_path = args.model_path+args.exper_name
    print(args.model_path)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    ## 这行代码设置 PyTorch 是否要检测梯度计算中的异常。
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    #print(args.config)
    #sys.exit()
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.test_set,
             args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")