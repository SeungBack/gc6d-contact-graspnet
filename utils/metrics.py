import logging
import open3d
import torch
import os

from options.config_utils import load_config
from models.losses import ContactGraspNetLoss


class Metrics(object):
    ITEMS = [
        {
        'name': 'ContactGraspNetLoss',
        'enabled': True,
        'eval_func': 'cls._get_contact_graspnet_loss_origin',
        'eval_object': ContactGraspNetLoss(load_config('model_params.yaml')),
        'is_greater_better': False,
        'init_value': 32767,
    }
    ]
    
    @classmethod
    def get(cls, pred, gt, require_emd=False):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            if not require_emd and 'emd' in item['eval_func']:
                _values[i] = torch.tensor(0.).to(gt.device)
            else:
                eval_func = eval(item['eval_func'])
                _values[i] = eval_func(pred, gt)
                
        return _values
        

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]
    
    @classmethod
    def _get_contact_graspnet_loss_origin(cls, pred, gt):
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, gt)
    
    @classmethod
    def _get_contact_graspnet_loss(cls, pred, gt):
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, gt)
    
    @classmethod
    def _get_single_grasp_chamfer_l2_distance(cls, pred, pred_sym, gt, gt_sym, quality, gt_quality, mesh):
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, pred_sym, gt, gt_sym, quality, gt_quality, mesh)
    
    @classmethod
    def _get_chamfer_l1_quality_collision(cls, pred, gt, quality, gt_quality, collision, collision_pc, grasp):
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, gt, quality, gt_quality, collision, collision_pc, grasp)
    
    @classmethod
    def _get_bi_grasp_chamfer_distance_v2(cls, pred, gt, quality, gt_quality):
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, gt, quality, gt_quality)
    
    @classmethod
    def _get_quality_acc(cls, pred_control_pt, quality, pred_grasps, mesh):
        quality_acc = cls.ITEMS[0]['eval_object']
        return quality_acc(pred_control_pt, quality, pred_grasps, mesh)
    
    @classmethod
    def _get_bi_grasp_chamfer_distance(cls, pred, gt, quality, gt_quality):
        chamfer_distance = cls.ITEMS[0]['eval_object']

        
    
    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))
        
    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())
    
    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value