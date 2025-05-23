import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
import configargparse
import sys

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add('--config', required=True, is_config_file=True, help='config file path')
    parser.add("--test_set", nargs="+", type=str, default=[])
    parser.add("--texts", nargs="+", type=str, default=[])
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 8],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}/language_features"
    output_dir = f"{dataset_path}/language_features_dim8"
    os.makedirs(output_dir, exist_ok=True)
    
    # copy the segmentation map
    for filename in os.listdir(data_dir):
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)


    checkpoint = torch.load(ckpt_path)
    eval_data = Autoencoder_dataset(data_dir, None)

    dataset = DataLoader(
        dataset=eval_data,
        batch_size=256,
        shuffle=False, 
        num_workers=4,
        drop_last=False   
    )



    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(dataset)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in eval_data.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v
