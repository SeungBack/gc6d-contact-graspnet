import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.utils as utils

class ContactGraspNetLoss(nn.Module):
    def __init__(self, config):
        super(ContactGraspNetLoss, self).__init__()
        self.config = config['MODEL']
        bin_weights = self.config['bin_weights']
        self.bin_vals = self._get_bin_vals()

        gripper_control_points = utils.get_control_point_tensor(1, use_torch=True, symmetric=False)
        gripper_control_points_sym = utils.get_control_point_tensor(1, use_torch=True, symmetric=True)
        self.register_buffer('gripper_control_points', gripper_control_points)
        self.register_buffer('gripper_control_points_sym', gripper_control_points_sym)
        self.register_buffer('bin_weights_tensor', torch.tensor(bin_weights, dtype=torch.float32))
        
        
    def forward(self, pred, data):
        """_summary_
        """
        # prediction values
        pred_grasps = pred['pred_grasps'] # (B, N, 4, 4)
        pred_scores = pred['pred_scores'] # (B, N, 1)
        pred_points = pred['pred_points'] # (B, N, 3)
        grasp_width_head = pred['grasp_width_head'].permute(0,2,1) # (B, N, len(bin_vals))
        
        # gt values
        pos_contact_points = data['pos_contact_points'] # (B, M, 3)
        pos_contact_width = data['pos_contact_width'] # (B, M)
        pos_contact_rot = data['pos_contact_rot'] # (B, M, 3, 3)
        pos_contact_trans = data['pos_contact_trans'] # (B, M, 3)
    
        
        width_label, grasp_success_label, contact_rot_label, contact_trans_label = self._compute_labels(pred_points=pred_points,
                                                                                                        pos_contact_points=pos_contact_points,
                                                                                                        pos_contact_width=pos_contact_width,
                                                                                                        pos_contact_rot=pos_contact_rot,
                                                                                                        pos_contact_trans=pos_contact_trans)

        min_geom_loss_divisor = float(1)
        pos_grasps_in_view = torch.clamp(grasp_success_label.sum(dim=1), min=min_geom_loss_divisor) # (B, )

        #* Grasp Confidence Loss
        # bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_label, reduction='none')  # B x N x 1
        # bin_ce_loss = torch.mean(bin_ce_loss)

        # balanced sampling for positive and negative grasps (sample pos number of neg grasps (highest loss ones) for each batch)
        bce_loss = F.binary_cross_entropy(pred_scores, grasp_success_label, reduction='none')  # B x N x 1
        pos_mask = grasp_success_label.bool()  # B x N x 1
        neg_mask = ~pos_mask  # B x N x 1
        pos_loss = bce_loss[pos_mask]  # Positive losses
        neg_loss = bce_loss[neg_mask]  # Negative losses
        neg_loss = torch.topk(neg_loss, k=pos_loss.shape[0], largest=True, sorted=False)[0]  # Select top-k hardest negatives
        bin_ce_loss = torch.cat([pos_loss, neg_loss], dim=0)
        bin_ce_loss = torch.mean(bin_ce_loss)

        #* Grasp Width Loss
        # convert to multihot
        bin_vals = self.config['offset_bins']
        grasp_width_labels_multihot = self._bin_labels_to_multihot(width_label, bin_vals) # (B, N, 10)
        width_loss = F.binary_cross_entropy_with_logits(grasp_width_head,
                                                        grasp_width_labels_multihot,
                                                        reduction='none') # (B, N, 1)
        bin_weights = self.bin_weights_tensor[None, None, :]
        width_loss = (bin_weights * width_loss).mean(axis=2)
        masked_width_loss = width_loss * grasp_success_label.squeeze(-1)
        width_loss = torch.mean(torch.sum(masked_width_loss, axis=1, keepdim=True) / pos_grasps_in_view)
        
        
        #* Grasp 6D Pose Loss
        # select positive grasp
        success_mask_rot = grasp_success_label.bool()[:, :, :, None] # (B, N, 1, 1)
        success_mask_rot = torch.broadcast_to(success_mask_rot, contact_rot_label.shape) # (B, N, 3, 3)
        pos_contact_rot_label = torch.where(success_mask_rot, contact_rot_label, torch.ones_like(contact_rot_label)*100000) # (B, N, 3, 3)
        success_mask_trans = grasp_success_label.bool() # (B, N, 1)
        success_mask_trans = torch.broadcast_to(success_mask_trans, contact_trans_label.shape) # (B, N, 3)
        pos_contact_trans_label = torch.where(success_mask_trans, contact_trans_label, torch.ones_like(contact_trans_label)*100000) # (B, N, 3)
        
        # gripper control points: broadcast over (B, N) without copying memory
        cp     = self.gripper_control_points.unsqueeze(1)     # (1, 1, 5, 3)
        sym_cp = self.gripper_control_points_sym.unsqueeze(1) # (1, 1, 5, 3)

        # transform control points: cp @ R^T + t  (equivalent to R @ cp^T, but avoids double permute)
        pred_contact_rot   = pred_grasps[:, :, :3, :3]  # (B, N, 3, 3)
        pred_contact_trans = pred_grasps[:, :, :3, 3]   # (B, N, 3)
        pred_control_points     = cp     @ pred_contact_rot.transpose(-1, -2) + pred_contact_trans.unsqueeze(2)      # (B, N, 5, 3)
        gt_control_points       = cp     @ pos_contact_rot_label.transpose(-1, -2) + pos_contact_trans_label.unsqueeze(2)  # (B, N, 5, 3)
        sym_gt_control_points   = sym_cp @ pos_contact_rot_label.transpose(-1, -2) + pos_contact_trans_label.unsqueeze(2)  # (B, N, 5, 3)

        # compute dist btw pred and gt control points
        expanded_pred_control_points     = pred_control_points.unsqueeze(2)     # (B, N, 1, 5, 3)
        expanded_gt_control_points       = gt_control_points.unsqueeze(1)       # (B, 1, N, 5, 3)
        expanded_sym_gt_control_points   = sym_gt_control_points.unsqueeze(1)   # (B, 1, N, 5, 3)

        # sum of squared dist btw all points
        squared_add     = torch.sum((expanded_pred_control_points - expanded_gt_control_points)**2,     dim=(3, 4))  # (B, N, N)
        sym_squared_add = torch.sum((expanded_pred_control_points - expanded_sym_gt_control_points)**2, dim=(3, 4))  # (B, N, N)

        # min over N gt grasps for each pred grasp, then take min between regular and symmetric
        # avoids creating (B, N, 2N) concat tensor
        squared_adds_k = torch.minimum(
            squared_add.min(dim=2, keepdim=True)[0],
            sym_squared_add.min(dim=2, keepdim=True)[0]
        )  # (B, N, 1)

        # mask negative grasp
        sum_grasp_success_labels = torch.sum(grasp_success_label, dim=2, keepdim=True) # (B, N, 1)
        binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1)
        min_adds = binary_grasp_success_labels * torch.sqrt(squared_adds_k) # (B, N, 1)
        if self.config.get('adds_use_pred_scores', False):
            adds_loss = torch.sum(pred_scores * min_adds, dim=1) / pos_grasps_in_view
        else:
            adds_loss = torch.sum(grasp_success_label * min_adds, dim=1) / pos_grasps_in_view
        adds_loss = torch.mean(adds_loss)
        total_loss = 1*bin_ce_loss + 1*width_loss + 10*adds_loss

        return total_loss, bin_ce_loss, width_loss, adds_loss
        
    
    def _get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config
        """
        bins_bounds = np.array(self.config['offset_bins'])
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1]) / 2
        bin_vals[-1] = bins_bounds[-1]
        bin_vals = np.minimum(bin_vals, self.config['gripper_width']-0.005)
        bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
        
        return bin_vals

    def _compute_labels(self, pred_points, pos_contact_points, pos_contact_width, pos_contact_rot, pos_contact_trans):
        """_summary_
        Project gt grasp labels on the predicted points
        All points w/o nearby successful grasp contact are considered negative contact points
        
        Args:
            pred_points (_type_): _description_
            pos_contact_points (_type_): _description_
            pos_contact_width (_type_): _description_
            pos_contact_rot (_type_): _description_
            pos_contact_trans (_type_): _description_

        Returns:
            _type_: _description_
        """
        nsample = 1
        radius = 0.005
        
        _, N, _ = pred_points.shape
        B, M, _ = pos_contact_points.shape
        
        # make grasp width B, M, 1
        pos_contact_width = pos_contact_width[:, :, None]
        
        
        # compute distance
        pred_points = pred_points.unsqueeze(2) # (B, N, 1, 3)
        pos_contact_points = pos_contact_points.unsqueeze(1) # (B, 1, M, 3)
        squared_dist = torch.sum((pred_points - pos_contact_points)**2, dim=-1) # (B, N, M)
        squared_dist_k, close_contact_pt_idcs = torch.topk(squared_dist, k=nsample, dim=2, largest=False, sorted=False) # (B, N, nsample)
        
        # group labels
        grouped_contact_width = utils.index_points(pos_contact_width, close_contact_pt_idcs) # (B, N, nsample, 1)
        grouped_contact_rot = utils.index_points(pos_contact_rot, close_contact_pt_idcs) # (B, N, nsample, 3, 3)
        grouped_contact_trans = utils.index_points(pos_contact_trans, close_contact_pt_idcs) # (B, N, nsample, 3)
        # compute label
        width_label = grouped_contact_width.mean(dim=2) # (B, N, 3)
        grasp_success_label = torch.mean(squared_dist_k, dim=2, keepdim=True) < radius**2 # (B, N, 1)
        grasp_success_label = grasp_success_label.type(torch.float32)
        grasp_contact_rot_label = grouped_contact_rot.mean(dim=2) # (B, N, 3, 3)
        grasp_contact_trans_label = grouped_contact_trans.mean(dim=2) # (B, N, 3)
        
        return width_label, grasp_success_label, grasp_contact_rot_label, grasp_contact_trans_label
        
    def _bin_labels_to_multihot(self, cont_labels, bin_boundaries):
        """_summary_
        Computes binned grasp width labels from continous labels and bin boundaries
        
        Args:
            garsp_width_labels (_type_): _description_
            bin_vals (_type_): _description_

        Returns:
            _type_: _description_
        """
        bins = []
        for b in range(len(bin_boundaries)-1):
            bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b+1])))
        multi_hot_labels = torch.cat(bins, dim=2)
        multi_hot_labels = multi_hot_labels.to(torch.float32).cuda()
        
        return multi_hot_labels


