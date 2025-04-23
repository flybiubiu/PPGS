import io
import os
import json
import sys
import torch
import torch.nn.functional as F
import torchvision
import cv2

import matplotlib.pyplot as plt

from PIL import Image
import seaborn as sns
import numpy as np
import colormaps
import sys
import mediapy as media
import imageio

from scene import Scene, GaussianModel
from scene.index_decoder import *
from gaussian_renderer import render

from utils.lem_utils import *
from utils.general_utils import safe_state

import configargparse
from arguments import ModelParams, PipelineParams, OptimizationParams
sys.path.append("..")
from autoencoder.model import Autoencoder


def colormap_saving(image: torch.Tensor, colormap_options, save_path = None):
    """
    if image's shape is (h, w, 1): draw colored relevance map;
    if image's shape is (h, w, 3): return directively;
    if image's shape is (h, w, c): execute PCA and transform it into (h, w, 3).
    """
    output_image = (
        colormaps.apply_colormap(
            image=image,
            colormap_options=colormap_options,
        ).cpu().numpy()
    )
    if save_path is not None:
        media.write_image(save_path, output_image, fmt="png")
    return output_image

def draw_rele_distrib(rele, kde=True):
    rele = rele.view(-1).detach().to("cpu").numpy()
    #print(rele.shape)(405080,)
    #print(rele.max())0.73873055
    
    plt.figure()
    if kde:
        sns.kdeplot(rele, color='blue', label='rele')
    else:
        plt.hist(rele, bins=30, color='blue', alpha=0.5, label='rele')
    plt.legend(loc='upper right')
    
    # create a file-like object from the figure, to convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close()

    return img

def smooth(mask):
    print("smooth")
    mask = mask.cpu()
    h, w = mask.shape[:2]
    im_smooth = mask.clone().cpu()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth

def rendering_mask(dataset, opt, pipe, checkpoint, codebook_pth, ae_ckpt_path,encoder_hidden_dims, decoder_hidden_dims, test_set, texts_dict, a, scale, com_type,is_smooth, is_fil_filter,device="cuda"):
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=False,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gaussians = GaussianModel(dataset.sh_degree, dataset.semantic_features_dim, dataset.points_num_limit)
    scene = Scene(dataset, gaussians, test_set=test_set, is_test=True)
    #index_decoder = IndexDecoder(dataset.semantic_features_dim, dataset.codebook_size).to(device)
    #print(checkpoint)#./output/mipnerf360/kitchen/0/chkpnt30000.pth
    (model_params, first_iter) = torch.load(checkpoint, map_location=torch.device(device))
    gaussians.restore(model_params, opt)
    ##index_decoder_ckpt = os.path.join(os.path.dirname(checkpoint), "index_decoder_" + os.path.basename(checkpoint))
    #index_decoder.load_state_dict(torch.load(index_decoder_ckpt))
    #print(codebook_pth)
    #/home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/kitchen/images/pyrclip_dino05_0000_32_896_1_20240730-110502_codebook.pt

    #codebook = read_codebook(codebook_pth)
    clip_rele = CLIPRelevance(device=device)
    #ae_ckpt_path =
    checkpoint_ae = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint_ae)
    model.eval()
    
    ouptut_dir = os.path.dirname(checkpoint)
    #print(ouptut_dir)./output/mipnerf360/kitchen/0
    
    eval_name = f"open_new_eval_{com_type}_s{scale}"
    gt_images_pth = f"{ouptut_dir}/{eval_name}/gt_images"
    pred_images_pth = f"{ouptut_dir}/{eval_name}/pred_images"
    pred_segs_pth = f"{ouptut_dir}/{eval_name}/pred_segs"
    rele_pth = f"{ouptut_dir}/{eval_name}/relevancy"
    
    os.makedirs(gt_images_pth, exist_ok=True)
    os.makedirs(pred_images_pth, exist_ok=True)
    os.makedirs(pred_segs_pth, exist_ok=True)
    os.makedirs(rele_pth, exist_ok=True)
    
    bg_color = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    
    viewpoint_stack = scene.getTestCameras().copy()
    for cam in viewpoint_stack:
        render_pkg = render(cam, gaussians, pipe, background)
        
        gt_image = cam.original_image
        image = render_pkg["render"].detach()
        
        torchvision.utils.save_image(gt_image, f"{gt_images_pth}/{cam.image_name}.png")
        torchvision.utils.save_image(image, f"{pred_images_pth}/{cam.image_name}.png")
        
        os.makedirs(f"{pred_segs_pth}/{cam.image_name}", exist_ok=True)
        os.makedirs(f"{pred_segs_pth}/{cam.image_name}/distr", exist_ok=True)
        #os.makedirs(f"{pred_segs_pth}_smooth/{cam.image_name}", exist_ok=True)
        #os.makedirs(f"{pred_segs_pth}_smooth/{cam.image_name}/distr", exist_ok=True)
        os.makedirs(f"{rele_pth}/{cam.image_name}/array", exist_ok=True)
        os.makedirs(f"{rele_pth}/{cam.image_name}/images", exist_ok=True)
        
        semantic_features = render_pkg["semantic_features"].detach().permute(1, 2, 0)
        #print(semantic_features.shape)torch.Size([8, 520, 779])
        h, w, _ = semantic_features.shape
        semantic_features = model.decode(semantic_features.flatten(0, 1))
        clip_features = semantic_features.view(h, w, -1)
        #print(clip_features.shape)torch.Size([821, 1236, 512])

        #norm_semantic_features = F.normalize(semantic_features, p=2, dim=0)
        #with torch.no_grad():
            #indices = index_decoder(norm_semantic_features.unsqueeze(0))
        #print(indices.shape)torch.Size([1, 32, 520, 779])
        #index_tensor = torch.argmax(indices, dim=1).squeeze()
        #print(index_tensor.shape)torch.Size([520, 779])
        #print(com_type)softmax
        '''
        if com_type == "argmax":
            # argmax
            clip_features = F.embedding(index_tensor, codebook[:, :512])
        elif com_type == "softmax":
            temp = 1   # ->0 = argmax, ->+inf = unifrom
            prob_tensor = torch.softmax(indices / temp, dim=1).permute(0, 2, 3, 1)  # (N, C=128, H, W)
            #print(prob_tensor.shape)torch.Size([1, 520, 779, 32])
            #print(codebook.shape)torch.Size([32, 896])
            #print(codebook[:, :512].shape)torch.Size([32, 512])
            clip_features = (prob_tensor @ codebook[:, :512]).squeeze()
            #print((prob_tensor @ codebook[:, :512]).shape)torch.Size([1, 520, 779, 512])
            #print(clip_features.shape)torch.Size([520, 779, 512])
        '''
        seg_shape = torch.rand(h, w)
        seg_indices = -1 * torch.ones_like(seg_shape)
        for i in range(len(list(texts_dict.keys()))):
            text = list(texts_dict.keys())[i]
            #print(type(texts_dict[text]))<class 'str'>
            #print(text)green grass
            #print(texts_dict[text][0])g
            #print(texts_dict[text][1])r
            if type(texts_dict[text]) is list:
                rele0 = clip_rele.get_relevancy(clip_features, texts_dict[text][0], scale).squeeze()[..., 0]
                rele1 = clip_rele.get_relevancy(clip_features, texts_dict[text][1], scale).squeeze()[..., 0]
                rele = torch.logical_or((rele0 >= a).float(), (rele1 >= a).float())
            else:
                rele = clip_rele.get_relevancy(clip_features, texts_dict[text], negatives=None, scale=scale).squeeze()[..., 0]
                p_i = torch.clip(rele - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6),colormaps.ColormapOptions("turbo"))
                mask = (rele < 0.5).squeeze()
                valid_composited_cpu = valid_composited.cpu()
                mask_cpu = mask.cpu()
                gt_image_cpu = gt_image
                gt_image_cpu = gt_image_cpu.permute(1, 2, 0)
                print(mask_cpu.shape)
                print(valid_composited_cpu.shape)
                print(gt_image.shape)
                valid_composited_cpu[mask_cpu, :] = gt_image_cpu[mask_cpu, :] * 1.0
                final_path = rele_pth + "/" +cam.image_name+"/" +texts_dict[text] + ".png"
                print(final_path)
                colormap_saving(valid_composited_cpu, colormap_options, final_path)
                if args.is_filter:
                    scale = 30
                    kernel = np.ones((scale, scale)) / (scale ** 2)
                    np_relev = rele.detach().cpu().numpy()
                    avg_filtered = cv2.filter2D(np_relev, -1, kernel)
                    avg_filtered = torch.from_numpy(avg_filtered).to(rele.device)
                    rele = 0.5 * (avg_filtered + rele)

                    output = rele
                    output = output - torch.min(output)
                    output = output / (torch.max(output) + 1e-9)
                    output = output * (1.0 - (-1.0)) + (-1.0)
                    rele = torch.clip(output, 0, 1)



            # norm
            # rele = (rele - rele.min()) / (rele.max() - rele.min())
            #print(rele.shape)torch.Size([520, 779])
            rele_distr_img = draw_rele_distrib(rele)
            print(a)
            msk = (rele >= a)
            if args.is_smooth:
                msk = smooth(msk)
            
            np.save(f"{rele_pth}/{cam.image_name}/array/{text}.npy", rele.detach().cpu().numpy())
            torchvision.utils.save_image(rele, f"{rele_pth}/{cam.image_name}/images/{text}.png")
            torchvision.utils.save_image(msk.float(), f"{pred_segs_pth}/{cam.image_name}/{text}.png")
            #torchvision.utils.save_image(msk_pred.float(), f"{pred_segs_pth}_smooth/{cam.image_name}/{text}.png")
            rele_distr_img.save(f"{pred_segs_pth}/{cam.image_name}/distr/{text}.png")
            
            seg_indices[msk] = i

        with open(f"{pred_segs_pth}/texts_dict.json", "w") as f:
            json.dump(texts_dict, f, indent=4)
        
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add('--exper_name', type=str, default="")
    parser.add_argument("--mode", type=str, default="search", choices=["search"])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--codebook", type=str, default = None)
    parser.add_argument("--test_set", nargs="+", type=str, default=[])
    parser.add_argument("--texts", nargs="+", type=str, default=[])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=100)
    parser.add_argument("--com_type", type=str, default="")
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[256, 128, 64, 32, 8],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument("--is_smooth", type=bool, default=False)
    parser.add_argument("--is_filter", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    print(args.start_checkpoint)
    args.start_checkpoint = args.start_checkpoint + args.exper_name + '/chkpnt30000.pth'
    print(args.start_checkpoint)
    print(args.alpha)
    print(args.is_smooth)
    print(args.is_filter)
    #sys.exit()
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    texts_dict = {}
    #print(args.texts)
    '''['LEGO Technic 856 Bulldozer', 'Basket Weave Cloth', 'Wood plat',
     'old pink striped cloth', 'Red Oven Gloves'] [11/08 23:02:06]'''
    for i in range(len(args.texts)):
        texts_dict[args.texts[i]] = args.texts[i]
    #ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "ae_ckpt/best_ckpt.pth")
    #print(args.ae_ckpt_dir)
    #/home/ps/Desktop/4t/cvpr/fegs/autoencoder/ckpt/room/best_ckpt.pth [04/10 15:43:51]
    rendering_mask(lp.extract(args), op.extract(args), pp.extract(args),
            args.start_checkpoint, args.codebook, args.ae_ckpt_dir,args.encoder_dims, args.decoder_dims,
            args.test_set, texts_dict, args.alpha, args.scale, args.com_type, args.is_smooth, args.is_filter)

    # All done
    print("Rendering done.")