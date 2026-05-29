import numpy as np
np.float = np.float64
import os
import argparse
from tqdm import tqdm
import torch
import h5py
import open3d as o3d

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.g1b import GraspNet1BLoader
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, transform_points
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.grasp import GraspGroup

MAX_GRASP_WIDTH = 0.14
NUM_VIEWS = 300
GRASP_HEIGHT = 0.02


def get_gripper_control_points():
    control_points = np.array([[-0.06, 0, 0],[-0.02, -0.04, 0], [-0.02, 0.04, 0],
                                [0.02, -0.04, 0], [0.02, 0.04, 0]])
    return control_points


def build_best_per_contact_group(g_points, g_offsets, fric_coefs, collision, pose, obj_idx, cam_pose_inv=None):
    """For each contact point, select the best (view, angle, depth) and return a GraspGroup.

    Args:
        g_points:    (Np, 3)         contact points in object frame
        g_offsets:   (Np, V, A, D, 3) offsets: [angle, depth, width]
        fric_coefs:  (Np, V, A, D)   friction coefficients
        collision:   (Np, V, A, D)   collision labels
        pose:        (4, 4)           object-to-world transform
        obj_idx:     int              object id
        cam_pose_inv:(4, 4) or None  world-to-camera transform (None if pose is already in camera frame)

    Returns:
        GraspGroup with at most one grasp per contact point
    """
    valid_mask = (fric_coefs > 0) & (~collision) & (g_offsets[..., 2] <= MAX_GRASP_WIDTH)
    scores = np.zeros_like(fric_coefs)
    scores[valid_mask] = 1.1 - fric_coefs[valid_mask]  # (Np, V, A, D)

    Np = scores.shape[0]

    # 1) best view per contact point
    view_score = scores.reshape(Np, NUM_VIEWS, -1).mean(axis=-1)  # (Np, V)
    best_view = np.argmax(view_score, axis=1)                      # (Np,)

    # 2) best angle for best view
    view_angle_score = scores[np.arange(Np), best_view, :, :]     # (Np, A, D)
    best_angle = np.argmax(view_angle_score.mean(axis=-1), axis=1) # (Np,)

    # 3) best depth
    depth_score = scores[np.arange(Np), best_view, best_angle, :]  # (Np, D)
    best_depth = np.argmax(depth_score, axis=1)                     # (Np,)

    final_score = scores[np.arange(Np), best_view, best_angle, best_depth]         # (Np,)
    final_offsets = g_offsets[np.arange(Np), best_view, best_angle, best_depth, :] # (Np, 3): angle, depth, width

    valid = final_score > 0
    if not valid.any():
        return GraspGroup()

    g_pts      = g_points[valid]
    scores_v   = final_score[valid]
    offsets_v  = final_offsets[valid]   # (Nv, 3)
    views_v    = best_view[valid]
    Nv = g_pts.shape[0]

    template_views = generate_views(NUM_VIEWS)
    views_sel = template_views[views_v]     # (Nv, 3)
    angles    = offsets_v[:, 0]             # (Nv,)
    depths    = offsets_v[:, 1]             # (Nv,)
    widths    = offsets_v[:, 2]             # (Nv,)

    Rs = batch_viewpoint_params_to_matrix(-views_sel, angles)       # (Nv, 3, 3)
    Rs = np.matmul(pose[:3, :3][np.newaxis], Rs)                    # apply object rotation
    if cam_pose_inv is not None:
        Rs = np.matmul(cam_pose_inv[:3, :3][np.newaxis], Rs)        # rotate to camera frame

    trans = transform_points(g_pts, pose)
    if cam_pose_inv is not None:
        trans = transform_points(trans, cam_pose_inv)

    rotations  = Rs.reshape(-1, 9)
    heights    = GRASP_HEIGHT * np.ones((Nv, 1))
    object_ids = obj_idx * np.ones((Nv, 1), dtype=np.int32)

    grasp_array = np.hstack([
        scores_v.reshape(-1, 1),
        widths.reshape(-1, 1),
        heights,
        depths.reshape(-1, 1),
        rotations,
        trans,
        object_ids
    ]).astype(np.float32)

    gg = GraspGroup()
    gg.grasp_group_array = grasp_array
    return gg


def load_scene_grasp_labels(g, scene_id, camera):
    """Load grasp labels once per scene using ann_id=0 to get the object list."""
    camera_poses = np.load(os.path.join(g.root, 'scenes', 'scene_%04d' % scene_id, camera, 'camera_poses.npy'))
    camera_pose = camera_poses[0]
    scene_reader = xmlReader(os.path.join(
        g.root, 'scenes', 'scene_%04d' % scene_id, camera, 'annotations', '0000.xml'))
    pose_vectors = scene_reader.getposevectorlist()
    obj_list, _ = get_obj_pose_list(camera_pose, pose_vectors)
    return g.loadGraspLabels(objIds=obj_list)


def main(cfgs):
    if cfgs.dataset_root is None:
        raise ValueError("--dataset_root is required (or set $G1B_ROOT)")
    if cfgs.n_workers < 1:
        raise ValueError("--n_workers must be >= 1")
    if cfgs.worker_id < 0 or cfgs.worker_id >= cfgs.n_workers:
        raise ValueError("--worker_id must be in [0, n_workers)")
    g = GraspNet1BLoader(cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)

    scene_ids = g.getSceneIds()
    start_id = (len(scene_ids) // cfgs.n_workers) * cfgs.worker_id
    end_id = (len(scene_ids) // cfgs.n_workers) * (cfgs.worker_id + 1) if cfgs.worker_id != cfgs.n_workers - 1 else len(scene_ids)
    scene_ids = sorted(scene_ids)[start_id:end_id]

    for scene_id in tqdm(scene_ids):
        collision_labels = g.loadCollisionLabels(scene_id)
        grasp_labels = load_scene_grasp_labels(g, scene_id, cfgs.camera)
        for ann_id in tqdm(range(256)):
            print('processing %06d_%06d' % (scene_id, ann_id))
            process_scene(cfgs, g, scene_id, ann_id, grasp_labels, collision_labels)


def process_scene(cfgs, g, scene_id, ann_id, grasp_labels, collision_labels):
    camera    = cfgs.camera
    split     = cfgs.split
    num_points = cfgs.num_points
    output_root = os.path.join(cfgs.dataset_root, 'scene_contacts')
    dist_thresh = 0.01
    batch_size  = 2048
    outlier     = 0.02

    output_path = os.path.join(output_root, camera, split, '%06d_%06d' % (scene_id, ann_id) + '.h5')
    if os.path.exists(output_path):
        return

    # Load camera pose and object list for this ann_id
    camera_poses = np.load(os.path.join(g.root, 'scenes', 'scene_%04d' % scene_id, camera, 'camera_poses.npy'))
    camera_pose  = camera_poses[ann_id]
    cam_pose_inv = np.linalg.inv(camera_pose)

    scene_reader = xmlReader(os.path.join(
        g.root, 'scenes', 'scene_%04d' % scene_id, camera, 'annotations', '%04d.xml' % ann_id))
    pose_vectors = scene_reader.getposevectorlist()
    obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)

    collision_dump = collision_labels['scene_' + str(scene_id).zfill(4)]

    scene_clouds    = g.loadScenePointCloud(scene_id, camera=camera, annId=ann_id, outlier=outlier)
    scene_clouds_ds = scene_clouds.voxel_down_sample(voxel_size=0.005)
    scene_points_ds = np.asarray(scene_clouds_ds.points)

    scene_points = np.asarray(scene_clouds.points)
    scene_colors = np.asarray(scene_clouds.colors)
    if scene_points.shape[0] > num_points:
        inds = np.random.choice(scene_points.shape[0], num_points, replace=False)
        scene_points = scene_points[inds]
        scene_colors = scene_colors[inds]

    scene_contact_points = []
    scene_grasp_dir      = []
    scene_grasp_app      = []
    scene_grasp_rot      = []
    scene_grasp_trans    = []
    scene_grasp_width    = []

    for i, (obj_idx, pose) in enumerate(zip(obj_list, pose_list)):
        g_points, g_offsets, fric_coefs = grasp_labels[obj_idx]
        collision = collision_dump[i]

        grasp_group = build_best_per_contact_group(
            g_points, g_offsets, fric_coefs, collision, pose, obj_idx, cam_pose_inv=cam_pose_inv
        )
        if len(grasp_group) == 0:
            continue

        # Filter by distance to scene surface
        grasp_points     = grasp_group.grasp_group_array[:, 13:16]
        grasp_points_gpu = torch.from_numpy(grasp_points).float().cuda()
        scene_points_gpu = torch.from_numpy(scene_points_ds).float().cuda()

        min_dists_list = []
        for j in range((grasp_points.shape[0] + batch_size - 1) // batch_size):
            start_idx = j * batch_size
            end_idx   = min((j + 1) * batch_size, grasp_points.shape[0])
            batch     = grasp_points_gpu[start_idx:end_idx]
            dists     = torch.norm(batch.unsqueeze(1) - scene_points_gpu.unsqueeze(0), p=1, dim=-1)
            min_dists_list.append(torch.min(dists, dim=1)[0])
            del dists
            torch.cuda.empty_cache()

        if len(min_dists_list) == 0:
            continue
        min_dists = torch.cat(min_dists_list, dim=0)
        valid_ids = torch.where(min_dists < dist_thresh)[0].cpu().numpy()
        grasp_group.grasp_group_array = grasp_group.grasp_group_array[valid_ids]

        if len(grasp_group) == 0:
            continue

        object_contact_points = []
        object_grasp_dir      = []
        object_grasp_app      = []
        object_grasp_rot      = []
        object_grasp_trans    = []
        object_grasp_width    = []

        for grasp in grasp_group:
            points   = grasp.translation
            rotation = grasp.rotation_matrix.reshape(3, 3)
            width    = grasp.width

            template_control_points = get_gripper_control_points()
            template_control_points[1:, 1] = np.sign(template_control_points[1:, 1]) * (width / 2)
            transformed_grasp = np.matmul(rotation, template_control_points.T).T + points

            mid_grasp_point = (transformed_grasp[1] + transformed_grasp[2]) / 2
            transformed_grasp = np.asarray([
                transformed_grasp[0],
                mid_grasp_point,
                transformed_grasp[1],
                transformed_grasp[2],
                transformed_grasp[3],
                transformed_grasp[4],
            ])

            finger_direction = transformed_grasp[5] - transformed_grasp[4]
            gripper_app = (transformed_grasp[1] - transformed_grasp[0]) / np.linalg.norm(transformed_grasp[1] - transformed_grasp[0])

            object_contact_points.append(points)
            object_grasp_dir.append(finger_direction / np.linalg.norm(finger_direction))
            object_grasp_app.append(gripper_app)
            object_grasp_rot.append(rotation)
            object_grasp_trans.append(points)
            object_grasp_width.append(width)

        scene_contact_points.extend(object_contact_points)
        scene_grasp_dir.extend(object_grasp_dir)
        scene_grasp_app.extend(object_grasp_app)
        scene_grasp_rot.extend(object_grasp_rot)
        scene_grasp_trans.extend(object_grasp_trans)
        scene_grasp_width.extend(object_grasp_width)

    scene_contact_points = np.asarray(scene_contact_points)
    scene_grasp_dir      = np.asarray(scene_grasp_dir)
    scene_grasp_app      = np.asarray(scene_grasp_app)
    scene_grasp_rot      = np.asarray(scene_grasp_rot)
    scene_grasp_trans    = np.asarray(scene_grasp_trans)
    scene_grasp_width    = np.asarray(scene_grasp_width)

    if cfgs.vis and len(scene_contact_points) > 0:
        visualize_result(scene_clouds, scene_contact_points, scene_grasp_rot,
                         scene_grasp_trans, scene_grasp_width, scene_id, ann_id)

    if not os.path.exists(os.path.join(output_root, camera, split)):
        os.makedirs(os.path.join(output_root, camera, split))
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('scene_contact_points', data=scene_contact_points)
        f.create_dataset('scene_grasp_dir',      data=scene_grasp_dir)
        f.create_dataset('scene_grasp_app',      data=scene_grasp_app)
        f.create_dataset('scene_grasp_rot',      data=scene_grasp_rot)
        f.create_dataset('scene_grasp_trans',    data=scene_grasp_trans)
        f.create_dataset('scene_grasp_width',    data=scene_grasp_width)
        f.create_dataset('scene_pc',             data=scene_points)
        f.create_dataset('scene_color',          data=scene_colors)


def visualize_result(scene_cloud, contact_points, grasp_rot, grasp_trans, grasp_width, scene_id, ann_id):
    """Visualize contact points and sampled grasps on the scene point cloud."""
    # Contact points as red spheres
    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector(contact_points)
    contact_pcd.paint_uniform_color([1, 0, 0])

    # Reconstruct GraspGroup for visualization (subsample to 50 for performance)
    M = len(contact_points)
    vis_inds = np.random.choice(M, min(M, 50), replace=False)
    rot_vis   = grasp_rot[vis_inds]     # (K, 3, 3)
    trans_vis = grasp_trans[vis_inds]   # (K, 3)
    width_vis = grasp_width[vis_inds]   # (K,)
    K = len(vis_inds)

    vis_array = np.hstack([
        np.ones((K, 1)),                      # score
        width_vis.reshape(-1, 1),             # width
        0.02 * np.ones((K, 1)),               # height
        0.02 * np.ones((K, 1)),               # depth
        rot_vis.reshape(-1, 9),               # rotation
        trans_vis,                            # translation
        -np.ones((K, 1), dtype=np.float32)   # obj_id
    ]).astype(np.float32)

    vis_group = GraspGroup()
    vis_group.grasp_group_array = vis_array

    geoms = [scene_cloud, contact_pcd] + vis_group.to_open3d_geometry_list()
    o3d.visualization.draw_geometries(
        geoms,
        window_name='scene %04d ann %04d  |  %d contact points  |  %d grasps shown' % (
            scene_id, ann_id, M, K)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                        default=os.environ.get('G1B_ROOT'),
                        help='Dataset root for GraspNet-1Billion (default: $G1B_ROOT)')
    parser.add_argument('--camera', default='realsense', choices=('realsense', 'kinect'),
                        help='realsense, kinect')
    parser.add_argument('--num_points', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--split', default='train', choices=('train', 'test'), help='train, test')
    parser.add_argument('--n_workers', type=int, default=1, help='number of workers to use')
    parser.add_argument('--worker_id', type=int, default=0, help='worker id (0~n_workers-1)')
    parser.add_argument('--vis', action='store_true', help='visualize result with open3d')
    cfgs = parser.parse_args()

    main(cfgs)
