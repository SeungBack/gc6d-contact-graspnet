import os
import h5py
from tqdm import tqdm
import torch
import numpy as np

from autolab_core import RigidTransform

import warnings
warnings.filterwarnings(action='ignore')
import torch.utils.data as data
from .build import DATASETS
import logging
logging.getLogger('autolab_core').setLevel(logging.ERROR)

GC6D_CAMERAS = ('realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid')
GC6D_CAMERA_BY_IMAGE_MOD = {
    0: 'zivid',
    1: 'realsense-d415',
    2: 'realsense-d435',
    3: 'azure-kinect',
}

@DATASETS.register_module()
class GraspClutter6D(data.Dataset):
    
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.num_positive_contacts = config.NUM_POSITIVE_CONTACTS
        self.camera = config.CAMERA
        self.subset = config.subset
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.caching = True
        self.cache = {}

    
    def make_dataset(self):
        files = []
        cameras = [camera.strip() for camera in self.camera.split(',')]
        invalid_cameras = [camera for camera in cameras if camera not in GC6D_CAMERAS]
        if invalid_cameras:
            raise ValueError(
                f"Invalid GraspClutter6D camera(s): {invalid_cameras}. "
                f"Choose from {GC6D_CAMERAS} or comma-separate multiple cameras.")
        if self.subset == 'train':
            contact_dir = os.path.join(self.data_root, 'scene_contacts', 'train')
            if not os.path.isdir(contact_dir):
                raise FileNotFoundError(
                    f"Preprocessed GraspClutter6D contacts not found: {contact_dir}. "
                    "Run scripts/preprocess_gc6d.py first.")
            train_files = os.listdir(contact_dir)
            for file in train_files:
                if not file.endswith('.h5'):
                    continue
                # realsense-d415: 1, realsense-d435: 2, azure-kinect: 3, zivid: 0
                idx = int(file.split('.')[0].split('_')[-1]) % 4
                if GC6D_CAMERA_BY_IMAGE_MOD[idx] in cameras:
                    files.extend([os.path.join('scene_contacts', 'train', file)])
        else:
            raise NotImplementedError(
                f"subset='{self.subset}' is not supported for GraspClutter6D training loader. "
                "Only 'train' is available.")
        if len(files) == 0:
            raise RuntimeError(
                f"No preprocessed GraspClutter6D samples found for subset='{self.subset}', camera='{self.camera}'.")
        print(f'Find {len(files)} samples for {self.subset} set with {self.camera} camera.')
        return files
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        

        for offset in range(self.size):
            path = self.paths[(index + offset) % self.size]
            scene_grasp_path = os.path.join(self.data_root, path)
            with h5py.File(scene_grasp_path, 'r') as f:
                scene_contact_points = f['scene_contact_points'][()]
                scene_grasp_dir = f['scene_grasp_dir'][()]
                scene_grasp_app = f['scene_grasp_app'][()]
                scene_grasp_rot = f['scene_grasp_rot'][()]
                scene_grasp_trans = f['scene_grasp_trans'][()]
                scene_grasp_width = f['scene_grasp_width'][()]
                scene_pc = f['scene_pc'][()]

            if scene_contact_points.ndim >= 2 and len(scene_contact_points) > 0:
                break
        else:
            raise RuntimeError(
                f"No valid GraspClutter6D contact labels found for subset='{self.subset}', camera='{self.camera}'.")

        pc_mean = np.mean(scene_pc, axis=0)
        scene_pc -= pc_mean
        scene_contact_points -= pc_mean
        scene_grasp_trans -= pc_mean
                
        if self.num_positive_contacts > len(scene_contact_points):
            sampled_contact_idcs = np.arange(len(scene_contact_points))
            sampled_contact_idcs_replacement = np.random.choice(
                np.arange(len(scene_contact_points)), 
                self.num_positive_contacts-len(scene_contact_points), replace=True)
            pos_sampled_contact_idcs = np.hstack([sampled_contact_idcs, sampled_contact_idcs_replacement])
        else:
            pos_sampled_contact_idcs = np.random.choice(
                np.arange(len(scene_contact_points)), 
                self.num_positive_contacts, 
                    replace=False)
        
        pos_contact_points = torch.from_numpy(scene_contact_points[pos_sampled_contact_idcs]).type(torch.float32)
        pos_contact_dirs = torch.from_numpy(scene_grasp_dir[pos_sampled_contact_idcs]).type(torch.float32)
        pos_contact_apps = torch.from_numpy(scene_grasp_app[pos_sampled_contact_idcs]).type(torch.float32)
        pos_grasp_rot = torch.from_numpy(scene_grasp_rot[pos_sampled_contact_idcs]).type(torch.float32)
        pos_grasp_trans = torch.from_numpy(scene_grasp_trans[pos_sampled_contact_idcs]).type(torch.float32)
        pos_grasp_width = torch.from_numpy(scene_grasp_width[pos_sampled_contact_idcs]).type(torch.float32)
        scene_pc = torch.from_numpy(scene_pc).type(torch.float32)
        
        data = dict(
            pc = scene_pc,
            pos_contact_points = pos_contact_points,
            pos_contact_dirs = pos_contact_dirs,
            pos_contact_apps = pos_contact_apps,
            pos_contact_rot = pos_grasp_rot,
            pos_contact_trans = pos_grasp_trans,
            pos_contact_width = pos_grasp_width,
            path = path
        )
        return data
