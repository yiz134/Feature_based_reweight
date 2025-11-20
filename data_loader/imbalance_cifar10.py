import sys
from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn.functional as F
import random
import json
import os
import copy


def fix_seed(seed=888):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_cifar10(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True, noise_file='', teacher_idx=None, seed=888):
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    if train:
        fix_seed(seed)
        train_idxs, val_idxs = train_val_split(base_dataset.targets, seed)

        train_dataset = CIFAR10_train(root, cfg_trainer, train_idxs, train=True,
                                      transform=transform_train, seed=seed)
        val_dataset = CIFAR10_val(root, cfg_trainer, val_idxs, train=train,
                                  transform=transform_val)

        if cfg_trainer.get('instance', False):
            train_dataset.instance_noise()
        elif cfg_trainer.get('asym', False):
            train_dataset.asymmetric_noise()
        elif cfg_trainer.get('imbalance_symmetric', False):
            # 新的 noise type：先长尾再 symmetric noise
            train_dataset.imbalance_symmetric_noise()
        elif cfg_trainer.get('noise_file', ''):
            train_dataset.train_labels_gt = train_dataset.train_labels.copy()
            noise_pack = torch.load(os.path.join(root, 'CIFAR-10_human.pt'))
            train_dataset.train_labels = np.array(
                noise_pack[cfg_trainer['noise_file']]
            )[train_dataset.indexs]
            actual_noise = (train_dataset.train_labels_gt != train_dataset.train_labels).mean()
            print("acutal noise=", actual_noise)
        else:
            train_dataset.symmetric_noise()

        print('##############')
        print(train_dataset.train_labels[:10])
        print(train_dataset.train_labels_gt[:10])

        if teacher_idx is not None:
            print(len(teacher_idx))
            train_dataset.truncate(teacher_idx)

        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        fix_seed(seed)
        train_dataset = []
        val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train,
                                  transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    if len(val_dataset) == 0:
        return train_dataset, None
    else:
        return train_dataset, val_dataset


def train_val_split(base_dataset: torchvision.datasets.CIFAR10, seed):
    fix_seed(seed)
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)  # 每类 4500 作为训练，其余 500 作为验证
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, seed=888):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        fix_seed(seed)
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]
        self.train_labels = np.array(self.targets)[indexs]
        self.indexs = np.array(indexs)
        self.prediction = np.zeros((len(self.train_data),
                                    self.num_classes,
                                    self.num_classes),
                                   dtype=np.float32)
        self.noise_indx = []
        self.seed = seed

    def make_imbalance(self):
        fix_seed(self.seed)
        num_classes = self.num_classes
        rho = float(self.cfg_trainer['imbalance_ratio'])  # e.g. 10, 50, 100

        labels = self.train_labels
        cls_counts = [np.sum(labels == c) for c in range(num_classes)]
        max_count = max(cls_counts)  

        cls_num_list = []
        for k in range(num_classes):
            n_k = max_count * (rho ** (-k / (num_classes - 1.0)))
            cls_num_list.append(int(n_k))

        selected_indices = []
        for c, num in enumerate(cls_num_list):
            idxs = np.where(labels == c)[0]
            np.random.shuffle(idxs)
            selected_indices.extend(idxs[:num])

        selected_indices = np.array(selected_indices)

        self.train_data = self.train_data[selected_indices]
        self.train_labels = self.train_labels[selected_indices]
        self.indexs = self.indexs[selected_indices]
        self.cls_num_list = cls_num_list  

        print("Imbalance (rho={}) cls_num_list: {}".format(rho, cls_num_list))

    def imbalance_symmetric_noise(self):
        self.make_imbalance()
        self.symmetric_noise()

    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        fix_seed(self.seed)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes,
                                                           dtype=np.int32)
        actual_noise = (self.train_labels_gt != self.train_labels).mean()
        print("acutal noise=", actual_noise)

    def asymmetric_noise(self):
        self.train_labels_gt = copy.deepcopy(self.train_labels)
        fix_seed(self.seed)

        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
        actual_noise = (self.train_labels_gt != self.train_labels).mean()
        print("acutal noise=", actual_noise)

    def instance_noise(self):
        '''
        Instance-dependent noise
        https://github.com/haochenglouis/cores/blob/main/data/utils.py
        '''
        self.train_labels_gt = copy.deepcopy(self.train_labels)
        fix_seed(self.seed)

        q_ = np.random.normal(loc=self.cfg_trainer['percent'], scale=0.1, size=int(1e6))
        q = []
        for pro in q_:
            if 0 < pro < 1:
                q.append(pro)
            if len(q) == 50000:
                break

        w = np.random.normal(loc=0, scale=1, size=(32 * 32 * 3, 10))
        for i, sample in enumerate(self.train_data):
            sample = sample.flatten()
            p_all = np.matmul(sample, w)
            p_all[self.train_labels_gt[i]] = -int(1e6)
            p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
            p_all[self.train_labels_gt[i]] = 1 - q[i]
            self.train_labels[i] = np.random.choice(np.arange(10),
                                                    p=p_all / sum(p_all))

    def truncate(self, teacher_idx):
        self.train_data = self.train_data[teacher_idx]
        self.train_labels = self.train_labels[teacher_idx]
        self.train_labels_gt = self.train_labels_gt[teacher_idx]

    def __getitem__(self, index):
        img, target, target_gt = (self.train_data[index],
                                  self.train_labels[index],
                                  self.train_labels_gt[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt

    def __len__(self):
        return len(self.train_data)


class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)
        self.train_labels_gt = self.train_labels.copy()

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.train_labels[idx] = np.random.randint(self.num_classes,
                                                           dtype=np.int32)

    def asymmetric_noise(self):
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7

    def instance_noise(self):
        '''
        Instance-dependent noise
        https://github.com/haochenglouis/cores/blob/main/data/utils.py
        '''
        q_ = np.random.normal(loc=self.cfg_trainer['percent'], scale=0.1, size=int(1e6))
        q = []
        for pro in q_:
            if 0 < pro < 1:
                q.append(pro)
            if len(q) == 50000:
                break

        w = np.random.normal(loc=0, scale=1, size=(32 * 32 * 3, 10))
        for i, sample in enumerate(self.train_data):
            sample = sample.flatten()
            p_all = np.matmul(sample, w)
            p_all[self.train_labels_gt[i]] = -int(1e6)
            p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
            p_all[self.train_labels_gt[i]] = 1 - q[i]
            self.train_labels[i] = np.random.choice(np.arange(10),
                                                    p=p_all / sum(p_all))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Returns:
            tuple: (image, target, index, clean_target)
        """
        img, target, target_gt = (self.train_data[index],
                                  self.train_labels[index],
                                  self.train_labels_gt[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt


from utils.parse_config import ConfigParser


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_batches=0, training=True,
                 num_workers=4, pin_memory=True, config=None,
                 teacher_idx=None, seed=888):
        if config is None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.data_dir = data_dir

        noise_file = '%sCIFAR10_%.1f_Asym_%s.json' % (
            config['data_loader']['args']['data_dir'],
            cfg_trainer['percent'],
            cfg_trainer['asym'])

        self.train_dataset, self.val_dataset = get_cifar10(
            config['data_loader']['args']['data_dir'],
            cfg_trainer,
            train=training,
            transform_train=transform_train,
            transform_val=transform_val,
            noise_file=noise_file,
            teacher_idx=teacher_idx,
            seed=seed
        )

        super().__init__(self.train_dataset, batch_size, shuffle,
                         validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)

    def run_loader(self, batch_size, shuffle, validation_split,
                   num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle,
                         validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)


def load_cifar10(config, num_clean_val, batch_size=256):
    train_loader = CIFAR10DataLoader(
        data_dir=config["data_loader"]["args"]["data_dir"],
        batch_size=128,  # 128
        shuffle=True,
        validation_split=0.0,
        num_workers=0,
        training=True,
        config=config,
        seed=config["trainer"]["seed"],
    )

    val_loader_full = train_loader.split_validation()
    val_ds = val_loader_full.dataset
    N = len(val_ds)

    k = int(num_clean_val)
    rng = np.random.default_rng(config["trainer"]["seed"])
    sel_idx = rng.choice(N, size=k, replace=False)
    sel_idx = np.sort(sel_idx)

    val_subset_A = Subset(val_ds, sel_idx)
    tmp_loader = torch.utils.data.DataLoader(
        val_subset_A, batch_size=k, shuffle=False, num_workers=0
    )
    X_val, y_val, _, y_gt = next(iter(tmp_loader))

    mask = np.ones(N, dtype=bool)
    mask[sel_idx] = False
    remain_idx = np.nonzero(mask)[0]
    val_subset_B = Subset(val_ds, remain_idx)
    val_loader = torch.utils.data.DataLoader(
        val_subset_B, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_dl = CIFAR10DataLoader(
        data_dir=config["data_loader"]["args"]["data_dir"],
        batch_size=256, shuffle=False, num_workers=0,
        training=False, config=config, seed=config["trainer"]["seed"],
    )
    test_loader = test_dl.split_validation()

    ds_tr = train_loader.train_dataset  # CIFAR10_train
    noisy_mask = (ds_tr.train_labels != ds_tr.train_labels_gt)
    train_noise_idx = torch.tensor(np.where(noisy_mask)[0])
    train_clean_idx = torch.tensor(np.where(~noisy_mask)[0])
    return train_loader, val_loader, test_loader, X_val.cuda(), y_val.cuda(), train_clean_idx, train_noise_idx
