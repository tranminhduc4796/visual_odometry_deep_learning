import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from spatialmath.base import eul2r, rpy2r, q2r

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def write_pred_traj(path, rot_pred, trans_pred):
    """
    Update the txt file storing trajectory value
    :param path: txt file to save trajectory array
    :param rot_pred: [1, seq_len, 3]
    :param trans_pred: [1, seq_len, 3]
    """
    # Only consider the first step
    rot_pred = rot_pred.detach().cpu().numpy()
    trans_pred = trans_pred.detach().cpu().numpy()
    rot_pred = rot_pred[0, 0, :]
    trans_pred = trans_pred[0, 0, :]

    pred_traj = np.concatenate((rot_pred, trans_pred)).reshape(1, 6)
    with open(path, 'a+') as f:
        f.write('\n')
        np.savetxt(f, pred_traj, fmt='%1.3f')


def get_gt_trajectory(vid_seq_id, vid_seq_len, datadir):
    gt_trans = np.empty([vid_seq_len, 3])
    gt_Rt = np.empty([vid_seq_len, 4, 4])
    poses = np.loadtxt(os.path.join(datadir, 'poses', str(vid_seq_id).zfill(2) + '.txt'))
    for frame in range(vid_seq_len):
        pose = np.concatenate((np.asarray(poses[frame]).reshape(3, 4), [[0., 0., 0., 1.]]), axis=0)
        gt_trans[frame, :] = pose[0:3, 3].T
        gt_Rt[frame, :] = pose
    return gt_trans, gt_Rt


def plot_seq(exp_dir, seq_idx, seq_len, traj_predicts, datadir, config):
    T = np.eye(4)
    trans_pred = np.empty([seq_len, 3])
    gt_trans, gt_poses = get_gt_trajectory(seq_idx, seq_len, datadir)

    # First frame as the world origin
    trans_pred[0] = np.zeros([1, 3])
    for frame_idx in range(seq_len - 1):
        # Relative pose: Transformation vector [rot, trans]
        pose2wrt1_traj_pred = traj_predicts[frame_idx, :]

        if config.outputParameterization == 'se3':
            trans_pred[frame_idx + 1] = pose2wrt1_traj_pred[:3]
        else:

            if config.outputParameterization == 'default':
                rel_R_pred = rpy2r(pose2wrt1_traj_pred[:3])
                rel_t_pred = np.reshape(pose2wrt1_traj_pred[3:], (3, 1))
            elif config.outputParameterization == 'euler':
                rel_R_pred = eul2r(pose2wrt1_traj_pred[0] / 10., pose2wrt1_traj_pred[1] / 10., pose2wrt1_traj_pred[2] / 10.)
                rel_t_pred = np.reshape(pose2wrt1_traj_pred[3:], (3, 1))
            elif config.outputParameterization == 'quaternion':
                rel_R_pred = q2r(np.transpose(pose2wrt1_traj_pred[:4]))
                rel_t_pred = np.reshape(pose2wrt1_traj_pred[4:], (3, 1))

            pose2wrt1_Rt_pred = np.concatenate((np.concatenate([rel_R_pred, rel_t_pred], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            T_abs = np.dot(T, pose2wrt1_Rt_pred)
            # Update the T matrix till now.
            T = T_abs

            # Get the origin of the frame (i+1)
            trans_pred[frame_idx + 1] = np.transpose(T[0:3, 3])

    "Get the ground truth camera trajectory"
    # Plot the estimated and ground-truth trajectories
    x_gt = gt_trans[:, 0]
    z_gt = gt_trans[:, 2]

    x_est = trans_pred[:, 0]
    z_est = trans_pred[:, 2]

    fig, ax = plt.subplots(1)
    ax.plot(x_gt, z_gt, 'c', label="ground truth")
    ax.plot(x_est, z_est, 'm', label="estimated")
    ax.legend()
    fig.savefig(os.path.join(exp_dir, 'plots', 'traj', str(seq_idx).zfill(2), 'traj'))


def plot_infer_seq(exp_dir, seq_len, traj_predicts, config):
    T = np.eye(4)
    trans_pred = np.empty([seq_len, 3])

    # First frame as the world origin
    trans_pred[0] = np.zeros([1, 3])
    for frame_idx in range(seq_len - 1):
        # Relative pose: Transformation vector [rot, trans]
        pose2wrt1_traj_pred = traj_predicts[frame_idx, :]

        if config.outputParameterization == 'se3':
            trans_pred[frame_idx + 1] = pose2wrt1_traj_pred[:3]
        else:

            if config.outputParameterization == 'default':
                rel_R_pred = rpy2r(pose2wrt1_traj_pred[:3])
                rel_t_pred = np.reshape(pose2wrt1_traj_pred[3:], (3, 1))
            elif config.outputParameterization == 'euler':
                rel_R_pred = eul2r(pose2wrt1_traj_pred[0] / 10., pose2wrt1_traj_pred[1] / 10., pose2wrt1_traj_pred[2] / 10.)
                rel_t_pred = np.reshape(pose2wrt1_traj_pred[3:], (3, 1))
            elif config.outputParameterization == 'quaternion':
                rel_R_pred = q2r(np.transpose(pose2wrt1_traj_pred[:4]))
                rel_t_pred = np.reshape(pose2wrt1_traj_pred[4:], (3, 1))

            pose2wrt1_Rt_pred = np.concatenate((np.concatenate([rel_R_pred, rel_t_pred], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            T_abs = np.dot(T, pose2wrt1_Rt_pred)
            # Update the T matrix till now.
            T = T_abs

            # Get the origin of the frame (i+1)
            trans_pred[frame_idx + 1] = np.transpose(T[0:3, 3])

    x_est = trans_pred[:, 0]
    z_est = trans_pred[:, 2]

    fig, ax = plt.subplots(1)
    ax.plot(x_est, z_est, 'm', label="estimated")
    ax.legend()
    fig.savefig(os.path.join(exp_dir, 'test_traj'))
