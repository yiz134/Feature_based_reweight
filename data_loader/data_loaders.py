import sys

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.cifar10 import get_cifar10
from data_loader.cifar100 import get_cifar100
from data_loader.clothing1m import get_clothing1m
#from data_loader.webvision import get_webvision
from utils.parse_config import ConfigParser
from PIL import Image


class Clothing1MDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4, pin_memory=True, config=None, teacher_idx=None, seed=8888):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])     

        self.data_dir = data_dir
        if config == None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        self.train_dataset, self.val_dataset = get_clothing1m(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=self.num_batches*self.batch_size, train=training,
#         self.train_dataset, self.val_dataset = get_clothing1m(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=260000, train=training,
                transform_train=self.transform_train, transform_val=self.transform_val, teacher_idx=teacher_idx, seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
        
        
class WebvisionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4, pin_memory=True, num_class=50, teacher_idx=None):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose([
                transforms.RandomCrop(227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.transform_val = transforms.Compose([
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         

        self.data_dir = data_dir
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        self.train_dataset, self.val_dataset = get_webvision(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=self.num_batches*self.batch_size, train=training,
                transform_train=self.transform_train, transform_val=self.transform_val, num_class=num_class, teacher_idx=teacher_idx)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)

