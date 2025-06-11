import torch
import torch.utils.data as torch_data
from scipy import io as scio
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
from PIL import Image


MNIST = 'mnist'
FASHION = 'fashion-mnist'
CIFAR = 'cifar'


def load_data(name, device=None):
    if device is None:
        device = torch.device('cpu')
    name = name.lower()
    if name == MNIST:
        transform_train = transform_test = torchvision.transforms.Compose([
            transforms.ToTensor(),
            #torchvision.transforms.Normalize(mean=[0.5],std=[0.5])
            ])
        train_dataset = torchvision.datasets.MNIST(root = "../dataset/mnist", transform=transform_train, train = True, download = True)
        test_dataset = torchvision.datasets.MNIST(root="../dataset/mnist", transform = transform_test, train = False, download = True)
        training_labels = np.array(train_dataset.targets)
        test_labels = np.array(test_dataset.targets)
        training_data = np.array(train_dataset.data)
        test_data = np.array(test_dataset.data)

    elif name == FASHION:
        transform_train = transform_test = torchvision.transforms.Compose([
            transforms.ToTensor(),
            #torchvision.transforms.Normalize(mean=[0.5],std=[0.5])
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root = "../dataset/fashion-mnist", transform=transform_train, train = True, download = True)
        test_dataset = torchvision.datasets.FashionMNIST(root="../dataset/fashion-mnist", transform = transform_test, train = False, download = True)
        training_labels = np.array(train_dataset.targets)
        test_labels = np.array(test_dataset.targets)
        training_data = np.array(train_dataset.data)
        test_data = np.array(test_dataset.data)
        

    elif name == CIFAR:
        transform_train = torchvision.transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = torchvision.transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10("../dataset/cifar-10", train=True, transform=transform_train, download=True)
        test_dataset = torchvision.datasets.CIFAR10("../dataset/cifar-10", train=False, transform=transform_test, download=True)
        # no transpose when depart
        training_labels = np.array(train_dataset.targets)
        test_labels = np.array(test_dataset.targets)
        training_data = train_dataset.data
        test_data = test_dataset.data
    else: 
        print("no dataset named {}".format(name))
    return training_data, training_labels, test_data, test_labels, transform_train, transform_test

def load_imagenet():
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_train = datasets.ImageFolder('../data/imagenet/images/train', transform_train)
    dataset_test = datasets.ImageFolder('../data/imagenet/images/val', transform_test)
    return dataset_train, dataset_test, 224, 1000, 3

class VPNDataset(torch_data.Dataset):
    def __init__(self, X, Y, transform=None):
        """
        Args:
            X: n * w * h * c
            Y: n
        """
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        img = self.X[index]
        target = self.Y[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.X)

    def get_feature_dim(self):
        return self.X.shape[-2]

    def get_data_size(self):
        return self.__len__()

    def get_n_class(self):
        return len(np.unique(self.Y))

    def get_n_channel(self):
        if self.X.shape[-1] == 3 :
            return 3
        else :
            return 1 

class TinyImageNet(Dataset):
    def __init__(self, train=True, transform=transforms.Compose([transforms.ToTensor()])):
        self.Train = train
        self.transform = transform
        self.train_dir = '../dataset/tiny-imagenet-200/train'
        self.val_dir = '../dataset/tiny-imagenet-200/val'

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = '../dataset/tiny-imagenet-200/words.txt'
        wnids_file = '../dataset/tiny-imagenet-200/wnids.txt'

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


