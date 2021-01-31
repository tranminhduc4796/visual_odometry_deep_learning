import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from spatialmath.base import eul2r, rpy2r, q2r

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def get_gt_trajectory(seq, seq_len, datadir):
    camera_traj = np.empty([seq_len, 3])
    full_traj = np.empty([seq_len, 4, 4])
    poses = np.loadtxt(os.path.join(datadir, 'poses', str(seq).zfill(2) + '.txt'))
    for frame in range(seq_len):
        pose = np.concatenate((np.asarray(poses[frame]).reshape(3, 4), [[0., 0., 0., 1.]]), axis=0)
        camera_traj[frame, :] = pose[0:3, 3].T
        full_traj[frame, :] = pose

    return camera_traj, full_traj


def plot_seq(exp_dir, seq, seq_len, trajectory, datadir, cmd, epoch):
    T = np.eye(4)
    estimated_camera_traj = np.empty([seq_len, 3])
    gt_camera_traj, fullT = get_gt_trajectory(seq, seq_len, datadir)

    # Extract the camera centres from all the frames

    # First frame as the world origin
    estimated_camera_traj[0] = np.zeros([1, 3])
    for frame in range(seq_len - 1):

        # Output is pose of frame i+1 with respect to frame i
        relativePose = trajectory[frame, :]

        if cmd.outputParameterization == 'se3':
            estimated_camera_traj[frame + 1] = relativePose[:3]
        else:

            if cmd.outputParameterization == 'default':
                R = rpy2r(np.transpose(relativePose[:3]))
                t = np.reshape(relativePose[3:], (3, 1))
            elif cmd.outputParameterization == 'euler':
                R = eul2r(relativePose[0] / 10., relativePose[1] / 10., relativePose[2] / 10., seq='xyz')
                t = np.reshape(relativePose[3:], (3, 1))
            elif cmd.outputParameterization == 'quaternion':
                R = q2r(np.transpose(relativePose[:4]))
                t = np.reshape(relativePose[4:], (3, 1))

            T_1 = fullT[frame, :]
            T_2 = fullT[frame + 1, :]
            T_r_gt = np.dot(np.linalg.inv(T_1), T_2)
            R_gt = T_r_gt[0:3, 0:3]
            t_gt = T_r_gt[0:3, 3].reshape(3, -1)

            T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            T_abs = np.dot(T, T_r)
            # Update the T matrix till now.
            T = T_abs

            # Get the origin of the frame (i+1), ie the camera center
            estimated_camera_traj[frame + 1] = np.transpose(T[0:3, 3])

    # Get the ground truth camera trajectory

    # Plot the estimated and ground-truth trajectories
    x_gt = gt_camera_traj[:, 0]
    z_gt = gt_camera_traj[:, 2]

    x_est = estimated_camera_traj[:, 0]
    z_est = estimated_camera_traj[:, 2]

    fig, ax = plt.subplots(1)
    ax.plot(x_gt, z_gt, 'c', label="ground truth")
    ax.plot(x_est, z_est, 'm', label="estimated")
    ax.legend()
    fig.savefig(os.path.join(exp_dir, 'plots', 'traj', str(seq).zfill(2), 'traj_' + str(epoch).zfill(3)))
