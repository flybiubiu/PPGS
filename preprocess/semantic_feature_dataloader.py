import json
from tqdm import tqdm
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import open_clip

from clip.clip_utils import *
from clip.dense_extractor.dense_extractor import DenseCLIPExtractorParams, DenseCLIPExtractor
from clip.pyramid.mixture_embedding_dataloader import MixtureEmbeddingDataloader
from dino.dino_dataloader import DinoExtractorParams, get_dinos

class SematicFeatureDataset(Dataset):
    def __init__(self, path, clip_params: DenseCLIPExtractorParams = None, dino_params: DinoExtractorParams = None):
        super().__init__()
        self.path = path
        self.clip_params = clip_params if clip_params else DenseCLIPExtractorParams()
        self.dino_params = dino_params if dino_params else DinoExtractorParams()
        
        self._concat_features()
    
    def _concat_features(self):
        clips = DenseCLIPExtractor(self.path, self.clip_params).data.to("cuda") # [194, 512, 106, 160]

        # stride4: [N, 384, 55, 83]
        # stride2: [N, 384, 109, 146]
        dinos = get_dinos(self.path, self.dino_params, half=True).permute(0, 3, 1, 2).to("cuda")
        
        # upsample dino feature map to clip feature map size
        h, w = clips.shape[2], clips.shape[3]
        if dinos.shape[0] < 300:
            sampled_dinos = F.interpolate(dinos, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # TO avoid RuntimeError: upsample_bilinear2d_nhwc only supports input tensors with less than INT_MAX elements
            sampled_dinos = torch.cat((F.interpolate(dinos[:200], size=(h, w), mode='bilinear', align_corners=False), 
                                       F.interpolate(dinos[200:], size=(h, w), mode='bilinear', align_corners=False)), 
                                      dim=0)
        
        del dinos
        torch.cuda.empty_cache()
        
        self.data = torch.cat((clips, sampled_dinos), dim=1)
        del clips, sampled_dinos
        
        self.data = self.data.contiguous().permute(0, 2, 3, 1)
        mask = torch.isnan(self.data) | torch.isinf(self.data)
        self.data[mask] = 0
        
    def _contract_feature(self):
        # read matches info from file
        with open("/media/nas/gsh/match_json/room_match.json", "r") as f:
            matches = json.loads(f.read())
        # contract
        for key in tqdm(matches.keys()):
            indices = torch.tensor(matches[key])
            n_indices = indices[:, 0].long()
            h_indices = indices[:, 1].long()
            w_indices = indices[:, 2].long()
            avg = self.data[n_indices, h_indices, w_indices, :512].mean(dim=0)
            self.data[n_indices, h_indices, w_indices, :512] = avg
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
    
    
class PyramidSematicFeatureDataset(Dataset):
    def __init__(self, path, clip_params: DenseCLIPExtractorParams = None, dino_params: DinoExtractorParams = None):
        super().__init__()
        self.path = path
        #print(self.path)/home/ps/Desktop/2t/LEGS/mipnerf360/kitchen/images
        #print(dino_params)None
        #self.dino_params = dino_params if dino_params else DinoExtractorParams()
        #print(self.dino_params)
        #DinoExtractorParams(model_type='dino_vits8', device='cuda', stride=4, load_size=224, layer=11, facet='key', bin=False, unflatten=True)

        self.device = "cuda"
        #model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        #model.eval()
        #self.model = model.to(self.device)
        '''
        self.process = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        '''
        self.image_paths = get_image_paths(path)
        #print(path)/home/ps/Desktop/2t/LEGS/mipnerf360/kitchen/images

        self.image_shape = to_tensor(Image.open(self.image_paths[0])).shape[1:3]
        #print(self.image_shape)torch.Size([2075, 3114])

        self.cfg = {
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(self.image_shape),
                "model_name": "ViT-B/16",
            }
                
        self.mixture_path = Path(os.path.join(path, f"pyramid_{self.cfg['tile_size_range'][0]}", "cache"))
        #print(self.mixture_path)/home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/kitchen/images/pyramid_0.05/cache

        self._concat_features()
    
    def _get_images(self):
        image_list = []
        for image_path in tqdm(self.image_paths):
            image_list.append(to_tensor(Image.open(image_path)))
        return torch.stack(image_list)
    
    def _load_mixture(self):
        mixture_data = None
        if self.mixture_path.with_suffix(".npy").exists():
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
            )
        else:
            image_list = self._get_images()
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                image_list = image_list,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
            )
        return mixture_data().to(self.device)
    
    def _concat_features(self):
        h, w = 60, 90
        feature_dir = os.listdir(os.path.join(self.path, "../language_features/"))
        feature_list = []
        for fn in feature_dir:
            if "_imgs_embeds.npy" in fn:
                feature_list.append(fn)
        feature_list.sort()
        feature_clips = []
        for i in range(len(feature_list)):
            #print(i)
            f_map_path = feature_list[i]
            f_map = torch.from_numpy(np.load(os.path.join(self.path, "../language_features/", f_map_path)))
            feature_clips.append(f_map.squeeze(0))
            #feature_clips.append(F.interpolate(f_map.permute(0, 3, 1, 2), size=(h, w), mode='nearest').squeeze(0))
        feature_clips = torch.stack(feature_clips).to(self.device)
        #print(feature_clips.dtype)
        '''
        clips = self._load_mixture().permute(0, 3, 1, 2)
        #print(clips.shape)torch.Size([311, 512, 22, 32])
        #print(clips.dtype)
        #print(clips.device)#cuda:0

        # stride4: [N, 384, 55, 83]
        # stride2: [N, 384, 109, 146]
        dinos = get_dinos(self.path, self.dino_params, half=True).permute(0, 3, 1, 2).to("cuda")
        #print(dinos.shape)torch.Size([311, 384, 55, 83])
        #print(dinos.dtype)torch.float32


        # upsample clip feature map to dino feature map size
        sampled_clips = F.interpolate(clips, size=(h, w), mode='nearest')
        #print(sampled_clips.dtype)torch.float16
        if dinos.shape[0] < 300:
            sampled_dinos = F.interpolate(dinos, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # TO avoid RuntimeError: upsample_bilinear2d_nhwc only supports input tensors with less than INT_MAX elements
            sampled_dinos = torch.cat((F.interpolate(dinos[:200], size=(h, w), mode='nearest'), 
                                       F.interpolate(dinos[200:], size=(h, w), mode='nearest')), 
                                      dim=0)
        
        del dinos
        torch.cuda.empty_cache()
        
        self.data = torch.cat((sampled_clips, sampled_dinos), dim=1)
        #print(self.data.shape)torch.Size([311, 896, 60, 90])
        #print(self.data.dtype)torch.float32
        del sampled_clips, sampled_dinos
        torch.cuda.empty_cache()
        '''

        self.data = feature_clips.contiguous().permute(0, 2, 3, 1).float()
        mask = torch.isnan(self.data) | torch.isinf(self.data)
        self.data[mask] = 0

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]