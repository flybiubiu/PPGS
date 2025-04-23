import os
import random
from tqdm import tqdm
from datetime import datetime   
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from quantizer import VectorQuantizer
from dino.dino_dataloader import DinoDataset
from clip.clip_dataloader import PyramidMixtureDataset, DenseMixtureDataset
from semantic_feature_dataloader import SematicFeatureDataset, PyramidSematicFeatureDataset
import sys
sys.path.append('..')
from utils.lem_utils import index_to_rgb_images, generate_colors
import configargparse

# Configuration
random.seed(0)

# Global writer initialization
writer = None

class Trainer:
    def __init__(self, args):
        self.args = args
        '''
        print(self.args)
        Namespace(base_codebook_path='', batch_size=32, beta=0.0, 
        config='configs/mipnerf360/garden.cfg', 
        dataset='mipnerf360', device='cuda', dino_weight=0.5, e_dim=896, 
        epoch=300, feat_type='pyrclip_dino', 
        image_dir='/home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/garden/images', 
        interv_n=20, kl_beta=1, load_balance_weight=1.0, max_p=0.0, min_p=0.0, n_e=128, 
        output_codebook_dir=None, shuffle=False, weight_mode='')
        '''
        self.tensorboard_step = 0
        self.writer = None
        self.prefix = self.name_prefix()
        #print(self.prefix)pyrclip_dino05_0000_128_896_1_20240713-005551
        self.initialize_writer()

    def initialize_writer(self):
        writer_dir_base = os.path.join("runs", self.args.dataset, self.args.image_dir.split("/")[-2], self.prefix)
        writer_dir = writer_dir_base
        if os.path.exists(writer_dir):
            counter = 0
            while os.path.exists(writer_dir):
                counter += 1
                writer_dir = f"{writer_dir_base}_{counter}"
        self.writer = SummaryWriter(log_dir=writer_dir)

    def name_prefix(self):
        dino_w_str = str(self.args.dino_weight).replace(".", "")
        #print(dino_w_str)05
        kl_beta_str = str(self.args.kl_beta).replace(".", "")#1
        min_p_str = str(self.args.min_p).replace(".", "")#00
        max_p_str = str(self.args.max_p).replace(".", "")#00
        # Format the current timestamp. For example: "20240331-235959"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.args.feat_type}{dino_w_str}_{self.args.weight_mode}{min_p_str}{max_p_str}_{self.args.n_e}_{self.args.e_dim}_{kl_beta_str}_{timestamp}"

    def select_dataset(self):
        dataset_cls = {
            'dino': DinoDataset,
            'pyrclip': PyramidMixtureDataset,
            'mixclip': DenseMixtureDataset,
            'clip_dino': SematicFeatureDataset,
            'pyrclip_dino': PyramidSematicFeatureDataset
        }.get(self.args.feat_type, DinoDataset)
        #print(dataset_cls)<class 'semantic_feature_dataloader.PyramidSematicFeatureDataset'>
        return dataset_cls(self.args.image_dir)

    def train(self):
        data_loader = DataLoader(self.select_dataset(), batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        color_map = generate_colors(self.args.n_e)
        model, optimizer, scheduler = self.setup_training()
        #print(data_loader)torch.Size([311, 55, 83, 384])

        model.train()
        for epoch in tqdm(range(self.args.epoch), dynamic_ncols=True):
            encoding_indices = []
            for feature in tqdm(data_loader, leave=False, dynamic_ncols=True):
                #print(feature.shape)orch.Size([32, 60, 90, 896])
                #print(*feature.shape[:3])32 60 90
                loss_cos, constrative_loss, loss_kl, encoding_indices_prob, d, z_q, perplexity, min_encodings, min_encoding_indices = model(feature)
                #print(min_encoding_indices.shape)orch.Size([172800, 1]) 32 * 60 * 90
                #print(min_encoding_indices.view(*feature.shape[:3], 1).shape)orch.Size([32, 60, 90, 1])
                encoding_indices.append(min_encoding_indices.view(*feature.shape[:3], 1))
                flattened_encoding_indices = min_encoding_indices.view(-1)
                #print(flattened_encoding_indices.shape)orch.Size([172800]
                #print(args.n_e)32
                #print(flattened_encoding_indices)ensor([ 5,  5, 14,  ...,  5,  5,  5], device='cuda:0')
                #print(flattened_encoding_indices.max())ensor(31, device='cuda:0')
                histogram = torch.histc(flattened_encoding_indices.float(), bins=args.n_e, min=0, max=args.n_e-1)
                #print(histogram.shape)orch.Size([32])
                ''''
                print(histogram)
                tensor([1.2710e+03, 1.0000e+00, 0.0000e+00, 1.0566e+04, 0.0000e+00, 1.9000e+01,:00<?, ?it/s]
                1.0300e+02, 0.0000e+00, 1.4367e+04, 5.0000e+01, 2.1700e+02, 7.8300e+02,
                1.3800e+02, 3.4000e+01, 6.1631e+04, 1.4200e+03, 0.0000e+00, 5.6088e+04,
                0.0000e+00, 9.5920e+03, 7.4000e+01, 4.5000e+01, 0.0000e+00, 9.9810e+03,
                2.5910e+03, 3.0460e+03, 7.4400e+02, 0.0000e+00, 3.9000e+01, 0.0000e+00,
                0.0000e+00, 0.0000e+00], device='cuda:0')
                '''
                num_elements = histogram.sum()
                #print(num_elements)tensor(172800., device='cuda:0')
                frac = histogram / num_elements#利用率 论文中的r
                #print(frac.shape)torch.Size([32])
                #print(encoding_indices_prob.shape)orch.Size([172800, 32])
                flattened_encoding_indices_prob = encoding_indices_prob.view(-1, args.n_e)
                #print(flattened_encoding_indices_prob.shape)torch.Size([172800, 32])
                #print(torch.mean(flattened_encoding_indices_prob, dim=0).shape)orch.Size([32])
                load_balancing_loss = (frac * torch.mean(flattened_encoding_indices_prob, dim=0)).sum()
                #print(d.shape)orch.Size([172800, 32])
                #print(d.mean())ensor(-0.0150, device='cuda:0', grad_fn=<MeanBackward0>) 
                loss_d = -1 * torch.log2(d.mean() if d.mean() > 0 else torch.tensor(1e-10))
                loss = loss_cos + args.load_balance_weight * load_balancing_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            #print(len(encoding_indices))9    279/32
            metric_loss = 1 - torch.mean(torch.cosine_similarity(feature[...,:512].to("cuda"), z_q[...,:512], dim = -1))
            encoding_indices_tensor = torch.cat(encoding_indices, dim=0).to("cpu")
            #print(encoding_indices_tensor.shape)orch.Size([279, 60, 90, 1])

            self.write_tensorboard(metric_loss, loss, loss_cos, loss_kl, load_balancing_loss, d, loss_d, perplexity)
            #print(self.args.interv_n)20
            if self.tensorboard_step % self.args.interv_n == 0:
                self.save_model(model, encoding_indices_tensor, color_map)
            self.tensorboard_step += 1
                

    def setup_training(self):
        #print(self.args.feat_type)pyrclip_dino
        concat = self.args.feat_type in ['clip_dino', 'pyrclip_dino']
        #print(concat) True
        model = VectorQuantizer(self.args.n_e, self.args.e_dim, self.args.beta, self.args.device, concat=concat, dino_weight=self.args.dino_weight)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=1000)
        return model, optimizer, scheduler

    def save_model(self, model, encoding_indices, color_map):
        output_dir = self.args.output_codebook_dir if self.args.output_codebook_dir else self.args.image_dir
        #print(output_dir)./data/mipnerf360/room/images
        torch.save(model.state_dict(), os.path.join(output_dir, f'{self.prefix}_clipnodino_codebook.pt'))
        torch.save(encoding_indices, os.path.join(output_dir, f'{self.prefix}_clipnodino_encoding_indices.pt'))
        for img_idx in range(0, 5):
            image = index_to_rgb_images(encoding_indices[img_idx].unsqueeze(0), color_map).permute(0, 3, 1, 2)[0]
            self.writer.add_image(f'encoding_image/pic{img_idx}', image, self.tensorboard_step)
        
    def write_tensorboard(self, metric_loss, loss, loss_cos, loss_kl, load_balancing_loss, d, loss_d, perplexity):
        # Example tensorboard writing function, extend as needed
        self.writer.add_scalar('loss/metric_loss', metric_loss.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss', loss.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss_cos', loss_cos.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss_kl', loss_kl.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/load_balancing_loss', load_balancing_loss.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/d', d.mean().item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss_d', loss_d.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/perplexity', perplexity.item(), self.tensorboard_step)

def parse_args():
    parser = configargparse.ArgParser(description="Training script parameters")
    parser.add_argument('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output_codebook_dir', type=str, default=None)
    parser.add_argument('--base_codebook_path', type=str, default="")
    parser.add_argument('--feat_type', type=str, default='dino')
    parser.add_argument('--dino_weight', type=float, default=0.1)
    parser.add_argument('--load_balance_weight', type=float, default=1.0)
    parser.add_argument('--n_e', type=int, default=128)
    parser.add_argument('--e_dim', type=int, default=512) # 384, 512, 512 + 384 = 896
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--kl_beta', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--interv_n', type=int, default=20)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_p', type=float, default=0.0)
    parser.add_argument('--weight_mode', type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
