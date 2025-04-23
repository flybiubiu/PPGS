#CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path ../data/bicycle  --encoder_dims 256 128 64 32 8 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name bicycle --config ../configs/bicycle.cfg
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path ../data/bonsai  --encoder_dims 256 128 64 32 8 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name bonsai --config ../configs/bonsai.cfg
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path ../data/counter  --encoder_dims 256 128 64 32 8 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name counter --config ../configs/counter.cfg
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path ../data/garden  --encoder_dims 256 128 64 32 8 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name garden --config ../configs/garden.cfg
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path ../data/kitchen  --encoder_dims 256 128 64 32 8 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name kitchen --config ../configs/kitchen.cfg
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset_path ../data/room  --encoder_dims 256 128 64 32 8 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name room --config ../configs/room.cfg

CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path ../data/bicycle --config ../configs/bicycle.cfg --dataset_name bicycle
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path ../data/bonsai --config ../configs/bonsai.cfg --dataset_name bonsai
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path ../data/counter --config ../configs/counter.cfg --dataset_name counter
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path ../data/garden --config ../configs/garden.cfg --dataset_name garden
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path ../data/kitchen --config ../configs/kitchen.cfg --dataset_name kitchen
CUDA_VISIBLE_DEVICES=1 python test.py --dataset_path ../data/room --config ../configs/room.cfg --dataset_name room