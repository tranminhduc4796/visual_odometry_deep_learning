"""
Main script: Train and test DeepVO on the KITTI odometry benchmark
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Other project files with definitions
import args
from KITTIDataset import KITTIDataset
from Model import DeepVO
from plotTrajectories import plot_seq
from Trainer import Trainer

# The following two lines are needed because, conda on Mila SLURM sets
# 'Qt5Agg' as the default version for matplotlib.use(). The interpreter
# throws a warning that says matplotlib.use('Agg') needs to be called
# before importing pyplot. If the warning is ignored, this results in
# an error and the code crashes while storing plots (after validation).
matplotlib.use('Agg')

# Parse commandline arguments
config = args.arguments

# Seed the RNGs (ensure deterministic outputs), if specified via commandline
if config.isDeterministic:
    # rn.seed(cmd.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

# Debug parameters. This is to run in 'debug' mode, which runs a very quick pass
# through major segments of code, to ensure nothing awful happens when we deploy
# on GPU clusters for instance, as a batch script. It is sometimes very annoying 
# when code crashes after a few epochs of training, while attempting to write a 
# checkpoint to a directory that does not exist.
if config.debug is True:
    config.debugIters = 3
    config.epochs = 2

# Set default tensor type to cuda.FloatTensor, for GPU execution
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Create directory structure, to store results
config.basedir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(config.basedir, config.cachedir, config.dataset)):
    os.makedirs(os.path.join(config.basedir, config.cachedir, config.dataset))

config.expDir = os.path.join(config.basedir, config.cachedir, config.dataset, config.expID)
if not os.path.exists(config.expDir):
    os.makedirs(config.expDir)
    print('Created dir: ', config.expDir)
if not os.path.exists(os.path.join(config.expDir, 'models')):
    os.makedirs(os.path.join(config.expDir, 'models'))
    print('Created dir: ', os.path.join(config.expDir, 'models'))
if not os.path.exists(os.path.join(config.expDir, 'plots', 'traj')):
    os.makedirs(os.path.join(config.expDir, 'plots', 'traj'))
    print('Created dir: ', os.path.join(config.expDir, 'plots', 'traj'))
if not os.path.exists(os.path.join(config.expDir, 'plots', 'loss')):
    os.makedirs(os.path.join(config.expDir, 'plots', 'loss'))
    print('Created dir: ', os.path.join(config.expDir, 'plots', 'loss'))
for seq in range(11):
    if not os.path.exists(os.path.join(config.expDir, 'plots', 'traj', str(seq).zfill(2))):
        os.makedirs(os.path.join(config.expDir, 'plots', 'traj', str(seq).zfill(2)))
        print('Created dir: ', os.path.join(config.expDir, 'plots', 'traj', str(seq).zfill(2)))

# Save all the command line arguments in a text file in the experiment directory.
cmdFile = open(os.path.join(config.expDir, 'args.txt'), 'w')
for arg in vars(config):
    cmdFile.write(arg + ' ' + str(getattr(config, arg)) + '\n')
cmdFile.close()

# TensorboardX visualization support
if config.tensorboardX is True:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(log_dir=config.expDir)

""" Model Definition + Weight init + FlowNet weight loading """

# Get the definition of the model
if config.modelType == 'flownet' or config.modelType is None:
    # Model definition without batchnorm

    deepVO = DeepVO(config.imageWidth, config.imageHeight, config.seqLen, 1, activation=config.activation,
                    parameterization=config.outputParameterization, \
                    batchnorm=False, dropout=config.dropout, flownet_weights_path=config.loadModel,
                    num_lstm_cells=config.num_lstm_cells)

deepVO.init_weights()
# CUDAfy
deepVO.cuda()
print('Loaded! Good to launch!')

""" Criterion, optimizer, and scheduler """

criterion = nn.MSELoss(reduction='sum')

if config.optMethod == 'adam':
    optimizer = optim.Adam(deepVO.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weightDecay,
                           amsgrad=False)
elif config.optMethod == 'sgd':
    optimizer = optim.SGD(deepVO.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weightDecay,
                          nesterov=False)
else:
    optimizer = optim.Adagrad(deepVO.parameters(), lr=config.lr, lr_decay=config.lrDecay, weight_decay=config.weightDecay)

# Initialize scheduler, if specified
if config.lrScheduler is not None:
    if config.lrScheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.lrScheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

"""  Main loop """

rotLosses_train = []
transLosses_train = []
totalLosses_train = []
rotLosses_val = []
transLosses_val = []
totalLosses_val = []
bestValLoss = np.inf
pNum = 0
for name, param in deepVO.named_parameters():
    pNum += 1
    if pNum <= 18:
        param.requires_grad = False

for epoch in range(config.epochs):

    print('================> Starting epoch: ' + str(epoch + 1) + '/' + str(config.epochs))

    # Create datasets for the current epoch
    # train_seq = [0, 1, 2, 8, 9]
    # train_startFrames = [0, 0, 0, 0, 0]
    # train_endFrames = [4540, 1100, 4660, 4070, 1590]
    # val_seq = [3, 4, 5, 6, 7, 10]
    # val_startFrames = [0, 0, 0, 0, 0]
    # val_endFrames = [800, 270, 2760, 1100, 1100, 1200]
    train_seq = [0]
    train_startFrames = [0]
    train_endFrames = [4500]
    val_seq = [0]
    val_startFrames = [0]
    val_endFrames = [4500]
    kitti_train = KITTIDataset(config.datadir, train_seq, train_startFrames, train_endFrames,
                               parameterization=config.outputParameterization,
                               width=config.imageWidth, height=config.imageHeight)
    kitti_val = KITTIDataset(config.datadir, val_seq, val_startFrames, val_endFrames,
                             parameterization=config.outputParameterization,
                             width=config.imageWidth, height=config.imageHeight)

    # Initialize a trainer (Note that any accumulated gradients on the model are flushed
    # upon creation of this Trainer object)
    trainer = Trainer(config, epoch, deepVO, kitti_train, kitti_val, criterion, optimizer,
                      scheduler=None)

    # weightb4 = []
    # for name, param in deepVO.named_parameters():
    # 	weightb4.append(param.data.clone())

    # Training loop
    print('===> Training: ' + str(epoch + 1) + '/' + str(config.epochs))
    startTime = time.time()
    rotLosses_train_cur, transLosses_train_cur, totalLosses_train_cur = trainer.train()
    print('Train time: ', time.time() - startTime)

    '''
    paramIt = 0
    for name,param in deepVO.named_parameters():
        weighta_i = param.data.clone()
        weightb_i = weightb4[paramIt]
        if weighta_i.equal(weightb_i):
            print("Weights not changed : ", name)
        else:
            print("Weights changed : ", name)
        paramIt+=1

    '''

    rotLosses_train += rotLosses_train_cur
    transLosses_train += transLosses_train_cur
    totalLosses_train += totalLosses_train_cur

    # Learning rate scheduler, if specified
    if config.lrScheduler is not None:
        scheduler.step()

    # Snapshot
    if config.snapshotStrategy == 'default':
        if epoch % config.snapshot == 0 or epoch == config.epochs - 1:
            print('Saving model after epoch', epoch, '...')
            torch.save(deepVO, os.path.join(config.expDir, 'models', 'model' + str(epoch).zfill(3) + '.pt'))
    elif config.snapshotStrategy == 'best' or 'none':
        # If we only want to save the best model, defer the decision
        pass

    # Validation loop
    print('===> Validation: ' + str(epoch + 1) + '/' + str(config.epochs))
    startTime = time.time()
    rotLosses_val_cur, transLosses_val_cur, totalLosses_val_cur = trainer.validate()
    print('Val time: ', time.time() - startTime)

    rotLosses_val += rotLosses_val_cur
    transLosses_val += transLosses_val_cur
    totalLosses_val += totalLosses_val_cur

    # Snapshot (if using 'best' strategy)
    if config.snapshotStrategy == 'best':
        if np.mean(totalLosses_val_cur) <= bestValLoss:
            bestValLoss = np.mean(totalLosses_val_cur)
            print('Saving model after epoch', epoch, '...')
            torch.save(deepVO, os.path.join(config.expDir, 'models', 'best' + '.pt'))

    # Save training curves
    fig, ax = plt.subplots(1)
    ax.plot(range(len(rotLosses_train)), rotLosses_train, 'r', label='rot_train')
    ax.plot(range(len(transLosses_train)), transLosses_train, 'g', label='trans_train')
    ax.plot(range(len(totalLosses_train)), totalLosses_train, 'b', label='total_train')
    ax.legend()
    plt.ylabel('Loss')
    plt.xlabel('Batch #')
    fig.savefig(os.path.join(config.expDir, 'loss_train_' + str(epoch).zfill(3)))

    fig, ax = plt.subplots(1)
    ax.plot(range(len(rotLosses_val)), rotLosses_val, 'r', label='rot_train')
    ax.plot(range(len(transLosses_val)), transLosses_val, 'g', label='trans_val')
    ax.plot(range(len(totalLosses_val)), totalLosses_val, 'b', label='total_val')
    ax.legend()
    plt.ylabel('Loss')
    plt.xlabel('Batch #')
    fig.savefig(os.path.join(config.expDir, 'loss_val_' + str(epoch).zfill(3)))

    # Plot trajectories (validation sequences)
    i = 0
    for s in val_seq:
        seqLen = val_endFrames[i] - val_startFrames[i]
        trajFile = os.path.join(config.expDir, 'plots', 'traj', str(s).zfill(2), \
                                'traj_' + str(epoch).zfill(3) + '.txt')
        if os.path.exists(trajFile):
            traj = np.loadtxt(trajFile)
            # traj = traj[:,3:]
            plot_seq(config.expDir, s, seqLen, traj, config.datadir, config, epoch)
        i += 1

print('Done !!')
