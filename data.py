import os
import torch
import torchvision
import torchvision.transforms as transforms


class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10

    class MNIST:
        class NoTransform:
            train = transforms.ToTensor()
            test = transforms.ToTensor()


def loaders(dataset, path, batch_size, num_workers, transform_name, scale=1.0, use_test=False,
            shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)
    num_classes = max(train_set.targets) + 1

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        if dataset == "MNIST":
            print("Using train (50000) + validation (10000)")
            train_set.data = train_set.data[:-10000]
            train_set.targets = train_set.targets[:-10000]

            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.data = test_set.data[-10000:]
            test_set.targets = test_set.targets[-10000:]
            # delattr(test_set, 'data')
            # delattr(test_set, 'targets')
        else:
            print("Using train (45000) + validation (5000)")
            train_set.train_data = train_set.train_data[:-5000]
            train_set.targets = train_set.targets[:-5000]

            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.test_data = test_set.train_data[-5000:]
            test_set.test_labels = test_set.targets[-5000:]
            delattr(test_set, 'train_data')
            delattr(test_set, 'targets')
    if scale < 1.0:
        train_set = torch.utils.data.Subset(train_set, torch.randperm(len(train_set))[:int(round(len(train_set)*scale))])
        test_set = torch.utils.data.Subset(test_set, torch.randperm(len(test_set))[:int(round(len(test_set)*scale))])

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, num_classes
