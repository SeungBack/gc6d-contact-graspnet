import argparse
import json
import os
import numpy as np
np.float = np.float64  # graspnetAPI uses deprecated np.float internally
import torch
import sys
import open3d as o3d

from utils.config import cfg_from_yaml_file
from utils import builder
from tqdm import tqdm

from graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
from graspclutter6dAPI import GraspClutter6D, GraspClutter6DEval
from utils.collision_detector import ModelFreeCollisionDetectorGPU

from data.g1b import GraspNet1BLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', help='pretrained weight')
    parser.add_argument('--model_config', type=str, default='configs/train_g1b.yaml')
    parser.add_argument('--camera', type=str, default='realsense')
    parser.add_argument('--test_dataset', default='g1b', choices=('g1b', 'gc6d'),
                        help='Test dataset to use [default: g1b]')
    parser.add_argument('--grasp_root', type=str, default=None,
                        help='Dataset root (default: env var of test dataset)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', type=str, default='test_seen')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--collision_thresh', type=float, default=0.0,
                        help='IoU threshold for collision detection (0 = skip)')
    parser.add_argument('--dump_dir', type=str, default=None,
                        help='Dump dir to save outputs (default: dump/<config_name>_<test_dataset>)')
    parser.add_argument('--infer', action='store_true', default=False, help='Run inference')
    parser.add_argument('--eval', action='store_true', default=False, help='Run evaluation')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize predictions with open3d (blocks per frame)')
    parser.add_argument('--arg_configs', nargs='*', default=[],
                        help='Override config values, e.g. MODEL.input_xyz_as_features:True')
    args = parser.parse_args()
    return args



def inference_single(base_model, scene_points, args, config, scene_id, im_id, max_width=0.10):
    replace = scene_points.shape[0] < 20000
    choice = np.random.choice(scene_points.shape[0], 20000, replace=replace)
    scene_points = scene_points[choice, :]

    pcd_torch = torch.from_numpy(scene_points).float()
    pcd_mean = torch.mean(pcd_torch, dim=0, keepdim=True)
    pcd_torch = pcd_torch - pcd_mean

    with torch.no_grad():
        pred = base_model(pcd_torch.unsqueeze(0).to(args.device))

    pred_grasps = pred['pred_grasps'].detach().cpu()  # (B, N, 4, 4)
    pred_grasps[:, :, :3, 3] += pcd_mean.squeeze()
    pred_scores = pred['pred_scores'].detach().cpu()   # (B, N)
    pred_grasp_width_bin = pred['pred_width'].detach().cpu()  # (B, N, 1)

    pred_contact_rot = pred_grasps[:, :, :3, :3].type(torch.double)
    pred_contact_trans = pred_grasps[:, :, :3, 3].type(torch.double)

    n_grasps = pred_scores.squeeze().shape[0]
    k = min(2048, n_grasps)
    sorted_pred_score, sorted_idx = torch.topk(pred_scores.squeeze(), k=k, largest=True)

    sorted_pred_contact_rot = pred_contact_rot[:, sorted_idx, :, :]
    sorted_pred_contact_trans = pred_contact_trans[:, sorted_idx, :]
    sorted_pred_width_bin = pred_grasp_width_bin[:, sorted_idx, :]

    sorted_pred_score = sorted_pred_score.numpy()
    sorted_pred_width_bin = sorted_pred_width_bin.numpy()[0, :, 0]
    sorted_pred_contact_rot = sorted_pred_contact_rot.numpy()[0]
    sorted_pred_contact_trans = sorted_pred_contact_trans.numpy()[0]

    widths = np.minimum(sorted_pred_width_bin * 1.2, max_width)  # (k,)
    rots = sorted_pred_contact_rot.reshape(k, 9)                  # (k, 9)
    g_array = np.column_stack([
        sorted_pred_score,             # score
        widths,                        # width
        np.full(k, 0.02),              # height
        np.full(k, 0.02),              # depth
        rots,                          # rotation (9)
        sorted_pred_contact_trans,     # translation (3)
        np.full(k, -1),                # object id
    ])

    gg = GraspGroup(g_array)

    if args.visualize:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_points))
        o3d.visualization.draw_geometries([pcd] + gg.to_open3d_geometry_list())

    save_dir = os.path.join(args.dump_dir, scene_id, args.camera)
    os.makedirs(save_dir, exist_ok=True)
    gg.save_npy(os.path.join(save_dir, im_id + '.npy'))


def run_inference(args, config):
    config.model.arg_configs = args.arg_configs or []
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    if args.test_dataset == 'gc6d':
        g = GraspClutter6D(root=args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        print('Inferencing on {} scenes'.format(len(scene_ids)))
        for scene_id in tqdm(scene_ids):
            for ann_id in tqdm(range(13), leave=False):
                im_id = g.annId2ImgId(ann_id, g.camera)
                scene_points, _ = g.loadScenePointCloud(scene_id, g.camera, ann_id, format='numpy', use_workspace=True)
                inference_single(base_model, scene_points, args, config,
                                 str(scene_id).zfill(6), str(im_id).zfill(6), max_width=0.14)
    else:
        g = GraspNet1BLoader(root=args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        print('Inferencing on {} scenes'.format(len(scene_ids)))
        for scene_id in tqdm(scene_ids):
            for ann_id in tqdm(range(256), leave=False):
                scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, outlier=0.02, format='numpy')
                inference_single(base_model, scene_points, args, config,
                                 'scene_' + str(scene_id).zfill(4), str(ann_id).zfill(4), max_width=0.10)


def apply_collision_to_dump(args):
    """Load saved npy files, apply soft (IoU) collision detection, write to {dump_dir}_cd/."""
    if args.collision_thresh <= 0:
        return args.dump_dir

    cd_dump_dir = args.dump_dir + '_cd'
    collision_device = torch.device(args.device.lower())

    if args.test_dataset == 'gc6d':
        g = GraspClutter6D(root=args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        for scene_id in tqdm(scene_ids, desc='Collision filtering'):
            for ann_id in range(13):
                im_id = g.annId2ImgId(ann_id, g.camera)
                scene_id_str = str(scene_id).zfill(6)
                im_id_str = str(im_id).zfill(6)
                src = os.path.join(args.dump_dir, scene_id_str, args.camera, im_id_str + '.npy')
                if not os.path.exists(src):
                    continue
                gg = GraspGroup().from_npy(src)
                scene_points, _ = g.loadScenePointCloud(scene_id, g.camera, ann_id, format='numpy', use_workspace=True)
                pts_gpu = torch.from_numpy(scene_points.astype(np.float32)).to(collision_device)
                collision_mask = ModelFreeCollisionDetectorGPU(pts_gpu, voxel_size=0.01).detect(
                    gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
                gg = gg[~collision_mask]
                dst_dir = os.path.join(cd_dump_dir, scene_id_str, args.camera)
                os.makedirs(dst_dir, exist_ok=True)
                gg.save_npy(os.path.join(dst_dir, im_id_str + '.npy'))
    else:
        g = GraspNet1BLoader(root=args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        for scene_id in tqdm(scene_ids, desc='Collision filtering'):
            for ann_id in range(256):
                scene_id_str = 'scene_' + str(scene_id).zfill(4)
                im_id_str = str(ann_id).zfill(4)
                src = os.path.join(args.dump_dir, scene_id_str, args.camera, im_id_str + '.npy')
                if not os.path.exists(src):
                    continue
                gg = GraspGroup().from_npy(src)
                scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, outlier=0.02, format='numpy')
                pts_gpu = torch.from_numpy(scene_points.astype(np.float32)).to(collision_device)
                collision_mask = ModelFreeCollisionDetectorGPU(pts_gpu, voxel_size=0.01).detect(
                    gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
                gg = gg[~collision_mask]
                dst_dir = os.path.join(cd_dump_dir, scene_id_str, args.camera)
                os.makedirs(dst_dir, exist_ok=True)
                gg.save_npy(os.path.join(dst_dir, im_id_str + '.npy'))

    return cd_dump_dir


def run_eval(args, dump_dir):
    if args.test_dataset == 'gc6d':
        ge = GraspClutter6DEval(root=args.grasp_root, camera=args.camera, split=args.split)
        res, ap = ge.eval_all(dump_dir, proc=args.num_workers)
        # ap = [mean_ap, ap0.4, ap0.8]
        eval_path = os.path.join(args.dump_dir, f'eval_results_{args.split}_{args.camera}.json')
        with open(eval_path, 'w') as f:
            json.dump({
                'split': args.split,
                'camera': args.camera,
                'ap': float(ap[0]),
                'ap_0.4': float(ap[1]),
                'ap_0.8': float(ap[2]),
            }, f, indent=2)
        print(f'Saved evaluation results to: {eval_path}')
        return res, ap

    ge = GraspNetEval(root=args.grasp_root, camera=args.camera, split=args.split)
    if args.split == 'test_seen':
        res, ap = ge.eval_seen(dump_dir, proc=args.num_workers)
    elif args.split == 'test_similar':
        res, ap = ge.eval_similar(dump_dir, proc=args.num_workers)
    elif args.split == 'test_novel':
        res, ap = ge.eval_novel(dump_dir, proc=args.num_workers)
    elif args.split == 'test':
        res, ap = ge.eval_all(dump_dir, proc=args.num_workers)
    else:
        raise ValueError(f'Unknown split: {args.split}')

    res_arr = np.array(res).reshape(-1, 6)
    res_mean = np.mean(res_arr, axis=0)
    print('----')
    print('AP0.4:', res_mean[1])
    print('AP0.8:', res_mean[3])
    print('AP:   ', np.mean(res_mean))
    print('----')

    eval_path = os.path.join(args.dump_dir, f'eval_results_{args.split}_{args.camera}.json')
    with open(eval_path, 'w') as f:
        json.dump({
            'split': args.split,
            'camera': args.camera,
            'ap': float(np.mean(res_mean)),
            'ap_0.4': float(res_mean[1]),
            'ap_0.8': float(res_mean[3]),
            'res_mean': res_mean.tolist(),
        }, f, indent=2)
    print(f'Saved evaluation results to: {eval_path}')
    return res, ap


GC6D_CAMERAS = ('realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid')
GC6D_CAMERA_ALIASES = {'realsense': 'realsense-d435', 'kinect': 'azure-kinect'}
G1B_CAMERAS = ('realsense', 'kinect')


def main():
    args = get_args()
    config = cfg_from_yaml_file(args.model_config, strict_env=False)

    if args.test_dataset == 'gc6d':
        args.camera = GC6D_CAMERA_ALIASES.get(args.camera, args.camera)
        if args.camera not in GC6D_CAMERAS:
            raise ValueError(f"Invalid camera '{args.camera}' for gc6d. Choose from: {GC6D_CAMERAS}")
    if args.test_dataset == 'g1b' and args.camera not in G1B_CAMERAS:
        raise ValueError(f"Invalid camera '{args.camera}' for g1b. Choose from: {G1B_CAMERAS}")

    if args.grasp_root is None:
        if args.test_dataset == 'gc6d':
            args.grasp_root = os.environ.get('GC6D_ROOT')
        else:
            args.grasp_root = os.environ.get('G1B_ROOT')
        if args.grasp_root is None:
            train_dataset_name = config.dataset.train._base_.NAME  # 'GraspNet1B' or 'GraspClutter6D'
            is_cross = (args.test_dataset == 'gc6d') != ('GraspClutter6D' in train_dataset_name)
            if is_cross:
                env_var = 'GC6D_ROOT' if args.test_dataset == 'gc6d' else 'G1B_ROOT'
                raise EnvironmentError(
                    f"Cross-dataset evaluation detected. Set ${env_var} or pass --grasp_root explicitly.")
            args.grasp_root = config.dataset.train._base_.DATA_PATH
            if isinstance(args.grasp_root, str) and '${' in args.grasp_root:
                env_var = 'GC6D_ROOT' if args.test_dataset == 'gc6d' else 'G1B_ROOT'
                raise EnvironmentError(
                    f"Set ${env_var}, pass --grasp_root, or edit {args.model_config} with an explicit DATA_PATH.")
    if args.dump_dir is None:
        config_stem = os.path.splitext(os.path.basename(args.model_config))[0]
        args.dump_dir = os.path.join('dump', f'{config_stem}_{args.test_dataset}')

    # Default: run both if neither flag is set
    run_infer = args.infer or (not args.infer and not args.eval)
    run_ev = args.eval or (not args.infer and not args.eval)

    if run_infer and args.model_checkpoint is None:
        raise ValueError("--model_checkpoint is required when running inference.")
    if args.test_dataset == 'gc6d' and args.split != 'test':
        raise ValueError("GC6D evaluation script supports only --split test.")

    if run_infer:
        run_inference(args, config)
    if run_ev:
        eval_dump_dir = apply_collision_to_dump(args)
        run_eval(args, eval_dump_dir)


if __name__ == '__main__':
    main()
