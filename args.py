import argparse

parser = argparse.ArgumentParser()

""" Model Options """
parser.add_argument('-loadFlowNet',
                    help='Whether or not to load pretrained weights. '
                         'If yes: then specify the path to the saved weights',
                    default=None)
parser.add_argument('-modelType',
                    help='Type of the model to be loaded : 1. deepVO |  2. flownet | 3. flownet_batchnorm',
                    type=str.lower,
                    choices=['deepvo', 'flownet', 'flownet_batchnorm'], default='flownet')
parser.add_argument('-initType', help='Weight initialization for the linear layers',
                    type=str.lower, choices=['xavier'], default='xavier')
parser.add_argument('-activation', help='Activation function to be used', type=str.lower,
                    choices=['relu', 'selu'], default='relu')
parser.add_argument('-dropout', help='Drop ratio of dropout at penultimate linear layer, if dropout is to be used.',
                    type=float, default=0.1)
parser.add_argument('-num_lstm_cells', help='Number of LSTM cells to stack together', type=int,
                    default=2)
parser.add_argument('-img_w', help='Width of the input image', type=int, default=1280)
parser.add_argument('-img_h', help='Height of the input image', type=int, default=384)

""" Dataset """
parser.add_argument('-dataset', help='dataset to be used for training the network', default='KITTI')
parser.add_argument('-outputParameterization', help='Parameterization of egomotion to be learnt by the network',
                    type=str.lower, choices=['default', 'quaternion', 'se3', 'euler'], default='default')

""" Hyper-parameters """
parser.add_argument('-batch_size', help='Number of samples in an iteration', type=int, default=2)
parser.add_argument('-lr', help='Learning rate', type=float, default=1e-5)
parser.add_argument('-momentum', help='Momentum', type=float, default=0.009)
parser.add_argument('-weight_decay', help='Weight decay', type=float, default=0.)
parser.add_argument('-lr_decay', help='Learning rate decay factor', type=float, default=0.)
parser.add_argument('-iterations', help='Number of iterations after loss is to be computed',
                    type=int, default=100)
parser.add_argument('-beta1', help='beta1 for ADAM optimizer', type=float, default=0.8)
parser.add_argument('-beta2', help='beta2 for ADAM optimizer', type=float, default=0.999)
parser.add_argument('-gradClip',
                    help='Max allowed magnitude for the gradient norm, '
                         'if gradient clipping is to be performed. (Recommended: 1.0)',
                    type=float)

parser.add_argument('-optMethod', help='Optimization method : adam | sgd | adagrad ',
                    type=str.lower, choices=['adam', 'sgd', 'adagrad'], default='adam')
parser.add_argument('-lrScheduler', help='Learning rate scheduler', default=None)

parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
parser.add_argument('-seq_len', help='Number of frames are involved to predict the poses at each time-steps',
                    type=int, default=3)

parser.add_argument('-scf', help='Scaling factor for the rotation loss terms',
                    type=float, default=100)
parser.add_argument('-gamma', help='For L2 regularization',
                    type=float, default=1.0)

""" Paths """
parser.add_argument('-cache_dir',
                    help='(Relative path to) directory in which to store logs, models, plots, etc.',
                    type=str, default='cache')
parser.add_argument('-datadir', help='Absolute path to the directory that holds the dataset',
                    type=str, default='./KITTI/dataset/')

""" Experiments, Snapshots, and Visualization """
parser.add_argument('-expID', help='experiment ID', default='tmp')
parser.add_argument('-snapshot', help='when to take model snapshots', type=int, default=5)
parser.add_argument('-snapshotStrategy',
                    help='Strategy to save snapshots. '
                         'Note that this has precedence over the -snapshot argument. '
                         '1. none: no snapshot at all | '
                         '2. default: as frequently as specified in -snapshot | '
                         '3. best: keep only the best performing model thus far',
                    type=str.lower, choices=['none', 'default', 'best'], default='best')
parser.add_argument('-tensorboardX', help='Whether or not to use tensorboardX for visualization',
                    type=bool, default=True)
parser.add_argument('-checkpoint', help='Model checkpoint to continue training',
                    default=None)

""" Debugging, Profiling, etc. """
parser.add_argument('-debug',
                    help='Run in debug mode, and execute 3 quick iterations per train loop. '
                         'Used in quickly testing whether the code has a silly bug.',
                    type=bool, default=False)
parser.add_argument('-profileGPUUsage', help='Profiles GPU memory usage and prints it every train/val batch', type=bool,
                    default=False)
parser.add_argument('-sbatch',
                    help='Replaces tqdm and print operations with file writes when True.'
                         ' Useful for reducing I/O when not running in interactive mode (eg. on clusters)',
                    type=bool, default=True)

""" Reproducibility """
parser.add_argument('-seed', help='Seed for pseudorandom number generator',
                    type=int, default=49)
parser.add_argument('-workers',
                    help='Number of threads available to the DataLoader',
                    type=int, default=1)

config = parser.parse_args()
