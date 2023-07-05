from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler


class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=4,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        if self.paths[index][-2:] == '_1': # dirty fix for rendering dataset twice
            total_view = 8
        else:
            total_view = 4
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
