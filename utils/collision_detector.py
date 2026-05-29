
import os
import sys
import numpy as np
import open3d as o3d
import torch

class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """
    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask3 = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask5 = (targets[:,:,1] < (widths/2 + self.finger_width))
        mask6 = (targets[:,:,1] > widths/2)
        # bottom mask
        mask7 = ((targets[:,:,0] <= depths - self.finger_length)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:,:,0] <= depths - self.finger_length - self.finger_width)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size**3)).reshape(-1)
        bottom_volume = (heights * (widths+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).reshape(-1)
        shifting_volume = (heights * (widths+2*self.finger_width) * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume+1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1)/inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume+1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume+1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume+1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume+1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value


class ModelFreeCollisionDetectorGPU():
    """ GPU-accelerated collision detection with identical logic to ModelFreeCollisionDetector.

        Input:
                scene_points: [torch.Tensor, (N,3), torch.float32] on GPU
                voxel_size: [float]
                    used for downsample
    """
    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        self.device = scene_points.device
        self.scene_points = self._voxel_down_sample(scene_points, voxel_size)

    def _voxel_down_sample(self, points, voxel_size):
        if points.shape[0] == 0:
            return points
        voxel_coords = torch.floor(points / voxel_size).long()
        _, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        num_unique = _.shape[0]
        summed = torch.zeros(num_unique, 3, device=self.device)
        summed.scatter_add_(0, inverse_indices.unsqueeze(-1).expand(-1, 3), points)
        counts = torch.bincount(inverse_indices).unsqueeze(-1).float()
        return summed / (counts + 1e-6)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05,
               return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Same interface and logic as ModelFreeCollisionDetector.detect().

            grasp_group: GraspGroup or (M,17) numpy array / torch.Tensor
        """
        if hasattr(grasp_group, 'grasp_group_array'):
            gg_array = torch.from_numpy(grasp_group.grasp_group_array).float().to(self.device)
        elif isinstance(grasp_group, np.ndarray):
            gg_array = torch.from_numpy(grasp_group).float().to(self.device)
        else:
            gg_array = grasp_group.float().to(self.device)

        approach_dist = max(approach_dist, self.finger_width)
        fw = self.finger_width
        fl = self.finger_length

        T       = gg_array[:, 13:16]                  # (M, 3)
        R       = gg_array[:, 4:13].reshape(-1, 3, 3) # (M, 3, 3)
        heights = gg_array[:, 2].unsqueeze(1)          # (M, 1)
        depths  = gg_array[:, 3].unsqueeze(1)
        widths  = gg_array[:, 1].unsqueeze(1)

        if T.shape[0] == 0:
            empty = np.array([], dtype=bool)
            ret_value = [empty]
            if return_empty_grasp:
                ret_value.append(empty)
            if return_ious:
                ret_value.append([np.array([], dtype=np.float32)] * 5)
            return ret_value if (return_empty_grasp or return_ious) else empty

        targets = self.scene_points.unsqueeze(0) - T.unsqueeze(1)  # (M, N, 3)
        targets = torch.einsum('bpi,bij->bpj', targets, R)

        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        mask2 = (targets[:, :, 0] > depths - fl)  & (targets[:, :, 0] < depths)
        mask3 = (targets[:, :, 1] > -(widths / 2 + fw))
        mask4 = (targets[:, :, 1] < -widths / 2)
        mask5 = (targets[:, :, 1] < (widths / 2 + fw))
        mask6 = (targets[:, :, 1] > widths / 2)
        mask7 = (targets[:, :, 0] <= depths - fl) & \
                (targets[:, :, 0] >  depths - fl - fw)
        mask8 = (targets[:, :, 0] <= depths - fl - fw) & \
                (targets[:, :, 0] >  depths - fl - fw - approach_dist)

        left_mask     = mask1 & mask2 & mask3 & mask4
        right_mask    = mask1 & mask2 & mask5 & mask6
        bottom_mask   = mask1 & mask3 & mask5 & mask7
        shifting_mask = mask1 & mask3 & mask5 & mask8
        global_mask   = left_mask | right_mask | bottom_mask | shifting_mask

        left_right_vol = (heights * fl * fw / self.voxel_size ** 3).squeeze(-1)
        bottom_vol     = (heights * (widths + 2 * fw) * fw / self.voxel_size ** 3).squeeze(-1)
        shifting_vol   = (heights * (widths + 2 * fw) * approach_dist / self.voxel_size ** 3).squeeze(-1)
        volume = left_right_vol * 2 + bottom_vol + shifting_vol

        global_iou     = global_mask.float().sum(dim=1) / (volume + 1e-6)
        collision_mask = (global_iou > collision_thresh).cpu().numpy()

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask]
        if return_empty_grasp:
            inner_mask = mask1 & mask2 & (~mask4) & (~mask6)
            inner_vol  = (heights * fl * widths / self.voxel_size ** 3).squeeze(-1)
            empty_mask = (inner_mask.float().sum(dim=1) / (inner_vol + 1e-6) < empty_thresh).cpu().numpy()
            ret_value.append(empty_mask)
        if return_ious:
            left_iou     = left_mask.float().sum(dim=1)     / (left_right_vol + 1e-6)
            right_iou    = right_mask.float().sum(dim=1)    / (left_right_vol + 1e-6)
            bottom_iou   = bottom_mask.float().sum(dim=1)   / (bottom_vol     + 1e-6)
            shifting_iou = shifting_mask.float().sum(dim=1) / (shifting_vol   + 1e-6)
            ret_value.append([
                global_iou.cpu().numpy(),
                left_iou.cpu().numpy(), right_iou.cpu().numpy(),
                bottom_iou.cpu().numpy(), shifting_iou.cpu().numpy(),
            ])
        return ret_value
