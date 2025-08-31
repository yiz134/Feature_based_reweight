import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import os
import json
from numpy.testing import assert_array_almost_equal

from utils.parse_config import ConfigParser
from torchvision import datasets, transforms
from base import BaseDataLoader

def fix_seed(seed=888):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def get_cifar100(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True, noise_file = '', teacher_idx=None, seed=888):
    base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)
    
    print (seed)
    
    if train:
        fix_seed(seed)
        train_idxs, val_idxs = train_val_split(base_dataset.targets, seed)
        
        train_dataset = CIFAR100_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train)
        val_dataset = CIFAR100_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            # if len(val_dataset) > 0:
            #     val_dataset.asymmetric_noise()
        else:
            train_dataset.symmetric_noise()
            # if len(val_dataset) > 0:
            #     val_dataset.symmetric_noise()
        
        if teacher_idx is not None:
            print(len(teacher_idx))
            train_dataset.truncate(teacher_idx)
        
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        fix_seed(seed)
        train_dataset = []
        val_dataset = CIFAR100_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    if len(val_dataset) == 0:
        return train_dataset, None
    else:
        return train_dataset, val_dataset

#     return train_dataset, val_dataset



def train_val_split(base_dataset: torchvision.datasets.CIFAR10, seed=888):
    fix_seed(seed)
    num_classes = 100
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
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

class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, seed=888):
        super(CIFAR100_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        fix_seed(seed)
        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]
        self.train_labels = np.array(self.targets)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.seed = seed
        #self.all_refs_encoded = torch.zeros(self.num_classes,self.num_ref,1024, dtype=np.float32)

        self.count = 0

    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        fix_seed(self.seed)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)
        actual_noise = (self.train_labels_gt != self.train_labels).mean()
        print("acutal noise=", actual_noise)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """
        fix_seed(self.seed)
        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

#     def build_for_cifar100(self, size, noise):
#         """ random flip between two random classes.
#         """
#         assert(noise >= 0.) and (noise <= 1.)

#         P = np.eye(size)
#         cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
#         P[cls1, cls2] = noise
#         P[cls2, cls1] = noise
#         P[cls1, cls1] = 1.0 - noise
#         P[cls2, cls2] = 1.0 - noise

#         assert_array_almost_equal(P.sum(axis=1), 1, 1)
#         return P
    def build_for_cifar100(self, size, noise):
        """ The noise matrix flips to the "next" class with probability 'noise'.
        """

        assert(noise >= 0.) and (noise <= 1.)

        P = (1. - noise) * np.eye(size)
        for i in np.arange(size - 1):
            P[i, i + 1] = noise

        # adjust last row
        P[size - 1, 0] = noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, asym=False, random_shuffle=False):
        self.train_labels_gt = self.train_labels.copy()
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5
        fix_seed(self.seed)
        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            print("actual noise=", actual_noise)
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy
            
    def truncate(self, teacher_idx):
        self.train_data = self.train_data[teacher_idx]
        self.train_labels = self.train_labels[teacher_idx]
        self.train_labels_gt = self.train_labels_gt[teacher_idx]    
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index],  self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt

    def __len__(self):
        return len(self.train_data)


class CIFAR100_val(torchvision.datasets.CIFAR100):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 100
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
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)


    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """
        
        print (P.shape[0] == P.shape[1])
        print (max(y) < P.shape[0])
        
        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert(noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5
        
        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy
    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt


class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,
                 num_workers=4, pin_memory=True, config=None, teacher_idx=None, seed=888):
        if config is None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        transform_train = transforms.Compose([
            # transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.data_dir = data_dir
        #         cfg_trainer = config['trainer']

        noise_file = '%sCIFAR100_%.1f_Asym_%s.json' % (
        config['data_loader']['args']['data_dir'], cfg_trainer['percent'], cfg_trainer['asym'])

        self.train_dataset, self.val_dataset = get_cifar100(config['data_loader']['args']['data_dir'], cfg_trainer,
                                                            train=training, transform_train=transform_train,
                                                            transform_val=transform_val, noise_file=noise_file,
                                                            teacher_idx=teacher_idx, seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)

    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)


def load_cifar100(config, num_clean_val, batch_size=128):
    train_loader = CIFAR100DataLoader(
        data_dir=config["data_loader"]["args"]["data_dir"],
        batch_size=128,
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

    test_dl = CIFAR100DataLoader(
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

# config = {
#     "data_loader": {"args": {"data_dir": "G:/datasets"}},
#     "trainer": {
#         "percent": 0.4,
#         "asym": True,
#         "instance": False,
#         "seed": 0,
#     },
# }
#
# train_loader, val_loader, test_loader, X_val, y_val, train_clean_idx, train_noise_idx = load_cifar100(config, 2000)
