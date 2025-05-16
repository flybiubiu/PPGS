#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/bicycle.cfg
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/bonsai.cfg
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/counter.cfg
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/garden.cfg
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/kitchen.cfg
#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/room.cfg


#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/bicycle.cfg
#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/bonsai.cfg
#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/counter.cfg
#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/garden.cfg
#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/kitchen.cfg
#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/room.cfg

#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/bicycle/0/open_new_eval_softmax_s10.0_a05/
#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/bonsai/0/open_new_eval_softmax_s10.0_a05/
#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/counter/0/open_new_eval_softmax_s10.0_a05/
#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/garden/0/open_new_eval_softmax_s10.0_a05/
#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/kitchen/0/open_new_eval_softmax_s10.0_a05/
#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/room/0/open_new_eval_softmax_s10.0_a05/


#cd sam_encoder
#CUDA_VISIBLE_DEVICES=0 python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h \
#      --input /home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/kitchen/images \
#      --output /home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/kitchen/sam_embeddings
#cd ..

#cd preprocess_ori
#CUDA_VISIBLE_DEVICES=0 python quantize_features.py --config configs/mipnerf360/kitchen.cfg \
#      --e_dim 768 --epoch 500
#cd ..

#CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mipnerf360/kitchen.cfg
#CUDA_VISIBLE_DEVICES=0 python render_mask.py --config configs/mipnerf360-rendering/kitchen.cfg
#CUDA_VISIBLE_DEVICES=0 python eval.py --path output/mipnerf360/kitchen/0/open_new_eval_softmax_s10.0_a05/


gpuid=0
name=room
exper_name=${name}_semantic4_final_supp_noembedding
#cd sam_encoder
#CUDA_VISIBLE_DEVICES=0 python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth \
#      --model-type vit_h \
#      --input /home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/${name}/images \
#      --output /home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/${name}/sam_embeddings
#cd ..

##cd preprocess_ori
##CUDA_VISIBLE_DEVICES=0 python quantize_features.py --config configs/mipnerf360/${name}.cfg \
      #--e_dim 1024 --epoch 300
#cd ..

CUDA_VISIBLE_DEVICES=${gpuid} python train.py --config configs/mipnerf360/${name}.cfg \
                                       --exper_name ${exper_name} --data_device cpu

CUDA_VISIBLE_DEVICES=${gpuid} python render_mask.py --config configs/mipnerf360-rendering/${name}.cfg  --data_device cpu\
                        --ae_ckpt_dir  ./autoencoder_dim8/ckpt/${name}/best_ckpt.pth \
                        --exper_name ${exper_name} \
                        --alpha 0.5 \
                        #--is_smooth True \
                        #--is_filter True

cp -r /home/ps/Desktop/2t/fegs_data/Mip-NeRF360_Dataset/${name}/segmentations ./output/mipnerf360/${name}/${exper_name}/open_new_eval_softmax_s10.0/

CUDA_VISIBLE_DEVICES=${gpuid} python eval.py --path output/mipnerf360/${name}/${exper_name}/open_new_eval_softmax_s10.0/

