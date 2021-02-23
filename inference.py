import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from args import config
from plotTrajectories import plot_infer_seq, write_pred_traj
from Model import DeepVO


# Class for providing an iterator for the KITTI visual odometry dataset
class KITTIInferenceDataset(Dataset):

    # Constructor
    def __init__(self, base_dir, sequence_len=3, parameterization='default',
                 width=1280, height=384):
        # Path to base directory of the KITTI odometry dataset
        # The base directory contains two directories: 'poses' and 'sequences'
        # The 'poses' directory contains text files that contain ground-truth pose
        # for the train sequences (00-10). The 11 train sequences and 11 test sequences
        # are present in the 'sequences' folder
        # Path to directory containing images
        self.imgDir = base_dir

        # Dimensions to be fed in the input
        self.width = width
        self.height = height
        self.channels = 3

        # n time-steps each time we evaluate
        self.seq_len = sequence_len

        # List of start frames and end frames for each sequence
        self.endFrames = len(os.listdir(self.imgDir))

        # Parameterization to be used to represent the transformation
        self.parameterization = parameterization

        # Create a dataframe to manage data
        self.df = self._init_dataframe()

    # Get dataset size
    def __len__(self):
        return len(self.df)

    # TO DO
    def __getitem__(self, idx):
        input_tensor_seq = []
        for frame1_idx in self.df.iloc[idx]:
            input_tensor = self.get_time_step_data(frame1_idx)
            input_tensor_seq.append(input_tensor)

        # Convert all item to Torch tensor
        input_tensor_seq = torch.stack(input_tensor_seq).float()
        return input_tensor_seq

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

    def _init_dataframe(self):
        """
        Create a dataframe which represents sequence for training.
        :return a dataframe with columns ['t', 't+1', ..., sequence_length, 'sequence_idx']
        """
        df = pd.DataFrame()
        df['t'] = [x for x in range(self.endFrames)]
        for step in range(self.seq_len - 1):
            df[f't+{step + 1}'] = df['t'].shift(-step - 1, fill_value=-1)
        return df[~(df == -1).any(axis=1)].reset_index(drop=True)

    def get_time_step_data(self, frame1_idx):
        """
        Get data of a time-step
        """
        frame2_idx = frame1_idx + 1

        # Directory containing images for the current sequence
        # Read in the corresponding images
        img1 = cv2.cvtColor(cv2.imread(os.path.join(self.imgDir, str(frame1_idx).zfill(6) + '.png')), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(os.path.join(self.imgDir, str(frame2_idx).zfill(6) + '.png')), cv2.COLOR_BGR2RGB)
        # Preprocess : returned after mean subtraction, resize and permute to N x C x W x H dims
        img1 = self.preprocess_img(img1)
        img2 = self.preprocess_img(img2)

        # Concatenate the images along the channel dimension
        pair = torch.cat((img1, img2), 0)
        tensor = pair.float()
        return tensor


def inference(exp_dir, dataset, model):
    best_path = os.path.join(exp_dir, 'model_best.pth.tar')
    print('Loading best weights...')
    state_dict = torch.load(best_path)['state_dict']
    model.load_state_dict(state_dict)
    print('Finish')
    model.eval()

    stat_bar = tqdm(enumerate(dataset), total=len(dataset), position=0, leave=True)
    traj_pred_path = os.path.join(exp_dir, 'test_traj.txt')
    with torch.no_grad():
        for idx, tensor in stat_bar:
            tensor = tensor.unsqueeze(0)

            # Load input to CUDA
            tensor = tensor.cuda(non_blocking=True)

            R_pred, t_pred = model.forward(tensor)

            R_pred = R_pred.permute(1, 0, 2)
            t_pred = t_pred.permute(1, 0, 2)

            write_pred_traj(traj_pred_path, R_pred, t_pred)

    seqLen = len(dataset)
    traj_predicts = np.loadtxt(traj_pred_path)
    plot_infer_seq(exp_dir, seqLen, traj_predicts, config)


if __name__ == '__main__':
    dataset = KITTIInferenceDataset('vinbdi_vid', sequence_len=2, width=448, height=192)
    exp_dir = 'infer'
    deepVO = DeepVO(448, 192, 2, 16)
    deepVO.cuda()
    inference(exp_dir, dataset, deepVO)
