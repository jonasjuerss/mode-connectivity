python train.py --dir=results/final/fmnist/c7-8 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --curve=PolyChain --num_bends=3 --init_start=results/final/fmnist/n7/checkpoint-200.pt --init_end=results/final/fmnist/n8/checkpoint-200.pt  --epochs=600 --lr=0.015 --wd=5e-4 --wandb_log --no_wandb --resume=results/final/fmnist/c7-8/checkpoint-220.pt

python train.py --dir=results/final/fmnist/n9 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --epochs=200 --lr=0.05 --wd=5e-4 --wandb_log --no_wandb
python train.py --dir=results/final/fmnist/n10 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --epochs=200 --lr=0.05 --wd=5e-4 --wandb_log --no_wandb

python train.py --dir=results/final/fmnist/c9-10 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --curve=PolyChain --num_bends=3 --init_start=results/final/fmnist/n9/checkpoint-200.pt --init_end=results/final/fmnist/n10/checkpoint-200.pt  --epochs=600 --lr=0.015 --wd=5e-4 --wandb_log --no_wandb

python train.py -system_end_points results/final/fmnist/n1/checkpoint-200.pt results/final/fmnist/n2/checkpoint-200.pt results/final/fmnist/n3/checkpoint-200.pt --dir=results/final/fmnist/system1-2-3 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --curve=PolyChainSystem --num_bends=3 --epochs=600 --lr=0.015 --wd=5e-4 --wandb_log --no_wandb
