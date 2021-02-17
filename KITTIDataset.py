from spatialmath.base import r2q, tr2eul, tr2rpy
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd


# Class for providing an iterator for the KITTI visual odometry dataset
class KITTIDataset(Dataset):

    # Constructor
    def __init__(self, kitti_base_dir, sequences=None, sequence_len=3, start_frames=None, end_frames=None,
                 parameterization='default', width=1280, height=384):

        # Path to base directory of the KITTI odometry dataset
        # The base directory contains two directories: 'poses' and 'sequences'
        # The 'poses' directory contains text files that contain ground-truth pose
        # for the train sequences (00-10). The 11 train sequences and 11 test sequences
        # are present in the 'sequences' folder
        self.baseDir = kitti_base_dir

        # Path to directory containing images
        self.imgDir = os.path.join(self.baseDir, 'sequences')
        # Path to directory containing pose ground-truth
        self.poseDir = os.path.join(self.baseDir, 'poses')

        # Max frames in each KITTI sequence
        self.KITTIMaxFrames = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]

        # Dimensions to be fed in the input
        self.width = width
        self.height = height
        self.channels = 3

        # n time-steps each time we evaluate
        self.seq_len = sequence_len

        # List of sequences that are part of the dataset
        # If nothing is specified, use sequence 1 as default
        self.sequences = sequences if sequences is not None else [1]

        # List of start frames and end frames for each sequence
        self.startFrames = start_frames if start_frames is not None else [0]
        self.endFrames = end_frames if end_frames is not None else [1100]

        # Parameterization to be used to represent the transformation
        self.parameterization = parameterization
        # Variable to hold length of the dataset
        self.length = 0

        # Verify input config
        self._verify()

        # Create a dataframe to manage data
        self.df = self._init_dataframe()

    # Get dataset size
    def __len__(self):
        return self.length

    # TO DO
    def __getitem__(self, idx):
        vid_seq_idx = self.df.iloc[idx, -1]

        input_tensor_seq = []
        R_seq = []
        t_seq = []
        for frame1_idx in self.df.iloc[idx, :-1]:
            input_tensor, R, t = self.get_time_step_data(frame1_idx, vid_seq_idx)

            input_tensor_seq.append(input_tensor)
            R_seq.append(R)
            t_seq.append(t)

        # Convert all item to Torch tensor
        input_tensor_seq = torch.stack(input_tensor_seq).float()
        R_seq = torch.cat(R_seq).float()
        t_seq = torch.cat(t_seq).float()
        return input_tensor_seq, R_seq, t_seq

    # Center and scale the image, resize and perform other preprocessing tasks
    def preprocess_img(self, img):
        # Flownet gets inputs in range [-0.5, 0.5]
        img = (img / 255.) - 0.5

        # Resize to the dimensions required
        img = np.resize(img, (self.height, self.width, self.channels))

        # Torch expects N,C,W,H
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        return img

    def get_ground_truth(self, seq_idx, frame1_idx, frame2_idx):
        poses = np.loadtxt(os.path.join(self.poseDir, str(seq_idx).zfill(2) + '.txt'), dtype=np.float32)
        pose1 = np.vstack([np.reshape(poses[frame1_idx].astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])
        pose2 = np.vstack([np.reshape(poses[frame2_idx].astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])
        # Compute relative pose from frame1 to frame2
        pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
        R = pose2wrt1[0:3, 0:3]
        t = (torch.from_numpy(pose2wrt1[0:3, 3]).view(-1, 3)).float()

        # Default parameterization: representation rotations as axis-angle vectors
        if self.parameterization == 'default':
            axisAngle = (torch.from_numpy(np.asarray(tr2rpy(R))).view(-1, 3)).float()
            return axisAngle, t
        # Quaternion parameterization: representation rotations as quaternions
        elif self.parameterization == 'quaternion':
            quaternion = np.asarray(r2q(R)).reshape((1, 4))
            quaternion = (torch.from_numpy(quaternion).view(-1, 4)).float()
            return quaternion, t
        # Euler parameterization: representation rotations as Euler angles
        elif self.parameterization == 'euler':
            rx, ry, rz = tr2eul(R)
            euler = (10. * torch.FloatTensor([rx, ry, rz]).view(-1, 3)).float()
            return euler, t

    def _verify(self):
        # Check if the parameters passed are consistent. Throw an error otherwise
        # KITTI has ground-truth pose information only for sequences 00 to 10
        if min(self.sequences) < 0 or max(self.sequences) > 10:
            raise ValueError('Sequences must be within the range [00-10]')
        if len(self.sequences) != len(self.startFrames):
            raise ValueError('There are not enough startFrames specified as there are sequences.')
        if len(self.sequences) != len(self.endFrames):
            raise ValueError('There are not enough endFrames specified as there are sequences.')
        # Check that, for each sequence, the corresponding start and end frames are within limits
        for i in range(len(self.sequences)):
            seq = self.sequences[i]
            if self.startFrames[i] < 0 or self.startFrames[i] > self.KITTIMaxFrames[seq]:
                raise ValueError('Invalid startFrame for sequence', str(seq).zfill(2))
            if self.endFrames[i] < 0 or self.endFrames[i] <= self.startFrames[i] or \
                    self.endFrames[i] > self.KITTIMaxFrames[seq]:
                raise ValueError('Invalid endFrame for sequence', str(seq).zfill(2))
            self.length += (self.endFrames[i] - self.startFrames[i])
        if self.length < 0:
            raise ValueError('Length of the dataset cannot be negative.')

    def _init_dataframe(self):
        """
        Create a dataframe which represents sequence for training.
        :return a dataframe with columns ['t', 't+1', ..., sequence_length, 'sequence_idx']
        """
        dataframes = []
        for idx, sequence_idx in enumerate(self.sequences):
            df = pd.DataFrame()
            df['t'] = [x for x in range(self.startFrames[idx], self.endFrames[idx])]
            for step in range(self.seq_len - 1):
                df[f't+{step+1}'] = df['t'].shift(-step-1, fill_value=-1)
            df['sequence_idx'] = sequence_idx
            dataframes.append(df)
        out_df = pd.concat(dataframes, axis=0, ignore_index=True)
        return out_df[~(out_df == -1).any(axis=1)].reset_index(drop=True)

    def get_time_step_data(self, frame1_idx, seq_idx):
        """
        Get data of a time-step
        """
        frame2_idx = frame1_idx + 1

        # Directory containing images for the current sequence
        curImgDir = os.path.join(self.imgDir, str(seq_idx).zfill(2), 'image_2')

        # Read in the corresponding images
        img1 = cv2.cvtColor(cv2.imread(os.path.join(curImgDir, str(frame1_idx).zfill(6) + '.png')), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(os.path.join(curImgDir, str(frame2_idx).zfill(6) + '.png')), cv2.COLOR_BGR2RGB)
        # Preprocess : returned after mean subtraction, resize and permute to N x C x W x H dims
        img1 = self.preprocess_img(img1)
        img2 = self.preprocess_img(img2)

        # Concatenate the images along the channel dimension
        pair = torch.cat((img1, img2), 0)
        tensor = pair.float()

        # Load pose ground-truth
        R, t = self.get_ground_truth(seq_idx, frame1_idx, frame2_idx)

        return tensor, R, t


