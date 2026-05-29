import os
import h5py
from tqdm import tqdm
import torch
import numpy as np

from autolab_core import RigidTransform
import open3d as o3d
import copy

from graspnetAPI import GraspNet
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.grasp import Grasp, GraspGroup
GRASP_HEIGHT = 0.02

import warnings
warnings.filterwarnings(action='ignore')
import torch.utils.data as data
from .build import DATASETS
import logging
logging.getLogger('autolab_core').setLevel(logging.ERROR)

G1B_CAMERAS = ('realsense', 'kinect')
G1B_SUBSETS = ('train', 'test')


def get_gripper_control_points():
    """Get control points for the gripper.

    """
    # default gripper width is 0.14m
    control_points = np.array([[-0.06, 0, 0],[-0.02, -0.04, 0], [-0.02, 0.04, 0],
                                [0.02, -0.04, 0], [0.02, 0.04, 0]])
    return control_points

class GraspNet1BLoader(GraspNet):
    def __init__(self, root, camera='realsense', split='train'):
        super(GraspNet1BLoader, self).__init__(root, camera=camera, split=split)
        
    def loadScenePointCloud(self, sceneId, camera, annId, align=False, format = 'open3d', outlier=0.02):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        - aligh: bool of whether align to the table frame.

        - format: string of the returned type. 'open3d' or 'numpy'

        - use_workspace: bool of whether crop the point cloud in the work space.

        - use_mask: bool of whether crop the point cloud use mask(z>0), only open3d 0.9.0 is supported for False option.
                    Only turn to False if you know what you are doing.

        - use_inpainting: bool of whether inpaint the depth image for the missing information.

        **Output:**

        - open3d.geometry.PointCloud instance of the scene point cloud.

        - or tuple of numpy array of point locations and colors.
        '''
        colors = self.loadRGB(sceneId = sceneId, camera = camera, annId = annId).astype(np.float32) / 255.0
        depths = self.loadDepth(sceneId = sceneId, camera = camera, annId = annId)
        seg = self.loadMask(sceneId = sceneId, camera = camera, annId = annId)
        intrinsics = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camK.npy'))
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        s = 1000.0
        
        xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        points_z = depths / s
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        points = np.stack([points_x, points_y, points_z], axis=-1)
        mask = (points_z > 0)
        seg = seg[mask]
        points = points[mask]
        colors = colors[mask]
        
        # workspace masking
        foreground = points[seg>0]
        xmin, ymin, zmin = foreground.min(axis=0)
        xmax, ymax, zmax = foreground.max(axis=0)
        mask_x = ((points[:,0] > xmin-outlier) & (points[:,0] < xmax+outlier))
        mask_y = ((points[:,1] > ymin-outlier) & (points[:,1] < ymax+outlier))
        mask_z = ((points[:,2] > zmin-outlier) & (points[:,2] < zmax+outlier))
        workspace_mask = (mask_x & mask_y & mask_z)
        
        points = points[workspace_mask]
        colors = colors[workspace_mask]
        


 
        if format == 'open3d':
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            return cloud
        elif format == 'numpy':
            return points, colors
        else:
            raise ValueError('Format must be either "open3d" or "numpy".')
        
    def loadGrasp(self, sceneId, annId=0, format = '6d', camera='kinect', grasp_labels = None, collision_labels = None, fric_coef_thresh=0.4):
        camera_poses = np.load(os.path.join(self.root,'scenes','scene_%04d' %(sceneId,),camera, 'camera_poses.npy'))
        camera_pose = camera_poses[annId]
        scene_reader = xmlReader(os.path.join(self.root,'scenes','scene_%04d' %(sceneId,),camera,'annotations','%04d.xml' %(annId,)))
        pose_vectors = scene_reader.getposevectorlist()

        obj_list,pose_list = get_obj_pose_list(camera_pose,pose_vectors)
        if grasp_labels is None:
            print('warning: grasp_labels are not given, calling self.loadGraspLabels to retrieve them')
            grasp_labels = self.loadGraspLabels(objIds = obj_list)
        if collision_labels is None:
            print('warning: collision_labels are not given, calling self.loadCollisionLabels to retrieve them')
            collision_labels = self.loadCollisionLabels(sceneId)

        num_views, num_angles, num_depths = 300, 12, 4
        template_views = generate_views(num_views)
        template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
        template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

        collision_dump = collision_labels['scene_'+str(sceneId).zfill(4)]

        # grasp = dict()
        grasp_group_list = []
        for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):

            sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
            collision = collision_dump[i]
            point_inds = np.arange(sampled_points.shape[0])

            num_points = len(point_inds)
            target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
            target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
            views = np.tile(template_views, [num_points, 1, 1, 1, 1])
            angles = offsets[:, :, :, :, 0]
            depths = offsets[:, :, :, :, 1]
            widths = offsets[:, :, :, :, 2]
    
            mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
            target_points = target_points[mask1]
            target_points = transform_points(target_points, trans)
            target_points = transform_points(target_points, np.linalg.inv(camera_pose))
            views = views[mask1]
            angles = angles[mask1]
            depths = depths[mask1]
            widths = widths[mask1]
            fric_coefs = fric_coefs[mask1]

            Rs = batch_viewpoint_params_to_matrix(-views, angles)
            Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)
            Rs = np.matmul(np.linalg.inv(camera_pose)[np.newaxis,:3,:3], Rs)

            num_grasp = widths.shape[0]
            scores = (1.1 - fric_coefs).reshape(-1,1)
            widths = widths.reshape(-1,1)
            heights = GRASP_HEIGHT * np.ones((num_grasp,1))
            depths = depths.reshape(-1,1)
            rotations = Rs.reshape((-1,9))
            object_ids = obj_idx * np.ones((num_grasp,1), dtype=np.int32)

            grasp_group_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(np.float32)
            grasp_group = GraspGroup()
            grasp_group.grasp_group_array = grasp_group_array
            # grasp_group.grasp_group_array = grasp_group.grasp_group_array[np.isclose(grasp_group.depths, 0.02)]
            grasp_group.grasp_group_array = grasp_group.grasp_group_array[grasp_group.widths <= 0.14] # filter out grasps that are too wide for the gripper
            grasp_group_list.append(copy.deepcopy(grasp_group))
        return grasp_group_list
    



@DATASETS.register_module()
class GraspNet1B(data.Dataset):
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
        if self.camera not in G1B_CAMERAS:
            raise ValueError(f"Invalid GraspNet-1Billion camera '{self.camera}'. Choose from {G1B_CAMERAS}.")
        if self.subset not in G1B_SUBSETS:
            raise ValueError(f"Invalid GraspNet-1Billion subset '{self.subset}'. Choose from {G1B_SUBSETS}.")
        if self.subset == 'train':
            contact_dir = os.path.join(self.data_root, 'scene_contacts', self.camera, 'train')
            if not os.path.isdir(contact_dir):
                raise FileNotFoundError(
                    f"Preprocessed GraspNet-1Billion contacts not found: {contact_dir}. "
                    "Run scripts/preprocess_g1b.py first.")
            train_files = os.listdir(contact_dir)
            for file in train_files:
                if file.endswith('.h5'):
                    files.extend([os.path.join('scene_contacts', self.camera, 'train', file)])
        else:
            contact_dir = os.path.join(self.data_root, 'scene_contacts', self.camera, 'test')
            if not os.path.isdir(contact_dir):
                raise FileNotFoundError(
                    f"Preprocessed GraspNet-1Billion contacts not found: {contact_dir}. "
                    "Run scripts/preprocess_g1b.py first.")
            test_files = os.listdir(contact_dir)
            for file in test_files:
                if file.endswith('.h5'):
                    files.extend([os.path.join('scene_contacts', self.camera, 'test', file)])
        if len(files) == 0:
            raise RuntimeError(
                f"No preprocessed GraspNet-1Billion samples found for subset='{self.subset}', camera='{self.camera}'.")
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
                f"No valid GraspNet-1Billion contact labels found for subset='{self.subset}', camera='{self.camera}'.")
            
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
        
        
        # # visualize scene_contact_points using open3d
        # import open3d as o3d
        # o3d_pc = o3d.geometry.PointCloud()
        # o3d_pc.points = o3d.utility.Vector3dVector(scene_pc.numpy())
        # g_array = []
        # for i in range(pos_contact_points.shape[0]):
        #     score = 1
        #     width = pos_grasp_width[i].item()
        #     rot = pos_grasp_rot[i].numpy().reshape(-1)
        #     trans = pos_contact_points[i].numpy().reshape(-1)
        #     g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
        # gg = GraspGroup(np.array(g_array))
        # print('Number of grasps to visualize:', len(gg))
        # # gg = gg.nms()
        # o3d.visualization.draw_geometries([o3d_pc] + gg.to_open3d_geometry_list())
        
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
        
        
