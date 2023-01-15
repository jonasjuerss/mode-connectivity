python train.py --dir=results/final/fmnist/n10 --resume=dir=results/final/fmnist/n10/checkpoint-160.pt --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --epochs=200 --lr=0.05 --wd=5e-4 --wandb_log --no_wandb

python train.py --dir=results/final/fmnist/c9-10 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --curve=PolyChain --num_bends=3 --init_start=results/final/fmnist/n9/checkpoint-200.pt --init_end=results/final/fmnist/n10/checkpoint-200.pt  --epochs=600 --lr=0.015 --wd=5e-4 --wandb_log --no_wandb

python train.py -system_end_points results/final/fmnist/n1/checkpoint-200.pt results/final/fmnist/n2/checkpoint-200.pt results/final/fmnist/n3/checkpoint-200.pt --dir=results/final/fmnist/system1-2-3 --dataset=FashionMNIST --use_test --transform=NoTransform --data_path=data --model=MNISTNet --curve=PolyChainSystem --num_bends=3 --epochs=600 --lr=0.015 --wd=5e-4 --wandb_log --no_wandb

python eval_curve_extensive.py --dir=results/final/fmnist/c1-2/eval --dataset=FashionMNIST --data_path=data --model=MNISTNet --wd=5e-4 --transform=NoTransform --curve=PolyChain --num_bends=3 --batch_size=128 --ckpt=results/final/fmnist/c1-2/checkpoint-600.pt --use_test 
python eval_curve_extensive.py --dir=results/final/fmnist/c3-4/eval --dataset=FashionMNIST --data_path=data --model=MNISTNet --wd=5e-4 --transform=NoTransform --curve=PolyChain --num_bends=3 --batch_size=128 --ckpt=results/final/fmnist/c3-4/checkpoint-600.pt --use_test 
python eval_curve_extensive.py --dir=results/final/fmnist/c5-6/eval --dataset=FashionMNIST --data_path=data --model=MNISTNet --wd=5e-4 --transform=NoTransform --curve=PolyChain --num_bends=3 --batch_size=128 --ckpt=results/final/fmnist/c5-6/checkpoint-600.pt --use_test 
python eval_curve_extensive.py --dir=results/final/fmnist/c7-8/eval --dataset=FashionMNIST --data_path=data --model=MNISTNet --wd=5e-4 --transform=NoTransform --curve=PolyChain --num_bends=3 --batch_size=128 --ckpt=results/final/fmnist/c7-8/checkpoint-600.pt --use_test 
python eval_curve_extensive.py --dir=results/final/fmnist/c9-10/eval --dataset=FashionMNIST --data_path=data --model=MNISTNet --wd=5e-4 --transform=NoTransform --curve=PolyChain --num_bends=3 --batch_size=128 --ckpt=results/final/fmnist/c9-10/checkpoint-600.pt --use_test 