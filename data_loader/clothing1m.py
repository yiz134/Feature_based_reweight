import numpy as np
import torch
import random
from torchvision import datasets, transforms
from base import BaseDataLoader
from utils.parse_config import ConfigParser
from PIL import Image
from collections import OrderedDict

def fix_seed(seed=777):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def get_clothing1m(root, cfg_trainer, num_samples=0, train=True,
                transform_train=None, transform_val=None, teacher_idx=None, seed=8888):

    if train:
        fix_seed(seed)
        train_dataset = Clothing1M_Dataset(root, cfg_trainer, num_samples=num_samples, train=train, transform=transform_train, seed=seed)
        val_dataset = Clothing1M_Dataset(root, cfg_trainer, val=train, transform=transform_val)
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")

    else:
        fix_seed(seed)
        train_dataset = []
        val_dataset = Clothing1M_Dataset(root, cfg_trainer, test= (not train), transform=transform_val)
        print(f"Test: {len(val_dataset)}")
        
    if teacher_idx is not None and train is True:
        print (len(teacher_idx))
        train_dataset.truncate(teacher_idx)

    return train_dataset, val_dataset

class Clothing1M_Dataset(torch.utils.data.Dataset):

    def __init__(self, root, cfg_trainer, num_samples=0, train=False, val=False, test=False, transform=None, num_class=14, seed=8888):
        
        fix_seed(seed)
        self.cfg_trainer = cfg_trainer
        self.root = root
        self.transform = transform
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}  

        self.train  = train
        self.val = val
        self.test = test

        with open('%s/annotations/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/annotations/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])  

        if train:          
            train_imgs=[]
            with open('%s/annotations/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for i , l in enumerate(lines):
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append((i,img_path)) 
            self.num_raw_example = len(train_imgs)                              
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            self.train_labels_ = []
            for id_raw, impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append((id_raw,impath))
                    self.train_labels_.append(int(label))
                    class_num[label]+=1
#                 else:
#                     print (label, class_num[label], (num_samples/14))
            random.shuffle(self.train_imgs)
            self.train_imgs = np.array(self.train_imgs)
            self.train_labels_ = np.array(self.train_labels_)

        elif test:
            self.test_imgs = []
            with open('%s/annotations/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)            
        elif val:
            self.val_imgs = []
            with open('%s/annotations/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.train:
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
        elif self.val:
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]   
        elif self.test:
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path] 
        image = Image.open(img_path).convert('RGB')
        if self.train:
            img0 = self.transform(image)
        
        if self.test or self.val:
            img = self.transform(image)
            return img, target, torch.as_tensor(index, dtype=torch.long), target
        else:
            return img0, target, torch.as_tensor(index, dtype=torch.long), target

    def __len__(self):
        if self.test:
            return len(self.test_imgs)
        if self.val:
            return len(self.val_imgs)
        else:
            return len(self.train_imgs) 

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath =  self.root + row[0]
                imlabel = float(row[1].replace('\n',''))
                imlist.append((impath, int(imlabel)))
        return imlist
    
    def truncate(self, teacher_idx):
        self.train_imgs = self.train_imgs[teacher_idx]
        self.train_labels_ = self.train_labels_[teacher_idx]


class Clothing1MDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,
                 num_workers=0, pin_memory=True, config=None, teacher_idx=None, seed=8888):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

        self.data_dir = data_dir
        # if config == None:
        #     config = ConfigParser.get_instance()
        # cfg_trainer = config['trainer']
        cfg_trainer = OrderedDict([
                                    ('epochs', 10),
                                    ('warmup', 0),
                                    ('save_dir', 'saved/'),
                                    ('save_period', 1),
                                    ('verbosity', 2),
                                    ('label_dir', 'saved/'),
                                    ('monitor', 'max test_my_metric'),
                                    ('early_stop', 2000),
                                    ('tensorboard', False),
                                    ('mlflow', True),
                                    ('_percent', 'Percentage of noise'),
                                    ('percent', 0.8),
                                    ('_begin', 'When to begin updating labels'),
                                    ('begin', 0),
                                    ('_asym', 'symmetric noise if false'),
                                    ('asym', False)])
        self.train_dataset, self.val_dataset = get_clothing1m(data_dir, cfg_trainer,
                                                              num_samples=self.num_batches * self.batch_size,
                                                              train=training,
                                                              #         self.train_dataset, self.val_dataset = get_clothing1m(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=260000, train=training,
                                                              transform_train=self.transform_train,
                                                              transform_val=self.transform_val, teacher_idx=teacher_idx,
                                                              seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)


def load_clothing1m(config, num_clean_val=2000):
    train_loader = Clothing1MDataLoader(
        config['data_dir'],
        batch_size=config['batch_size'],
        shuffle=True,
        validation_split=0.0,
        num_batches=config['num_batches'],
        training=True,
        num_workers=0,
        pin_memory=True,
        seed=config['seed']
    )

    val_loader   = train_loader.split_validation()

    test_loader = Clothing1MDataLoader(
        config['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=0,
        seed=config['seed']
    ).split_validation()

    X_list, y_list, got = [], [], 0
    with torch.no_grad():
        for imgs, labels, _, _ in val_loader:
            need = num_clean_val - got
            if need <= 0:
                break
            X_list.append(imgs[:need])
            y_list.append(labels[:need])
            got += min(imgs.size(0), need)

    if X_list:
        X_val = torch.cat(X_list, dim=0).cuda(non_blocking=True)
        y_val = torch.cat(y_list, dim=0).cuda(non_blocking=True)
    else:
        X_val = torch.empty(0, device='cuda')
        y_val = torch.empty(0, dtype=torch.long, device='cuda')

    return train_loader, val_loader, test_loader, X_val, y_val

# config = {
#   "batch_size": 64,
#   "num_batches": 2000,
#     "data_dir": "G:/datasets/Clothing1M/clothing1M"
# }
# train_loader, val_loader, test_loader, X_val, y_val = load_clothing1m(config)
# for img, target_noisy, id_raw, target_also_noisy in train_loader:
#     print(target_noisy == target_also_noisy)
# print(len(train_loader))