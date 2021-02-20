import os

from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from args import config
from helpers import init_dir_structure, save_checkpoint
from KITTIDataset import KITTIDataset
from Model import DeepVO
import matplotlib.pyplot as plt
from plotTrajectories import plot_seq, write_pred_traj


def main():
    # Set the progress to be deterministic (reproducible)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.set_deterministic(True)

    # Debug mode
    if config.debug is True:
        config.epochs = 2

    # Set default tensor type to cuda.FloatTensor, for GPU execution
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create directory structure, to store results
    base_dir = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(base_dir, config.cache_dir, config.dataset, config.expID)
    init_dir_structure(config, base_dir, exp_dir)

    # Save all the command line arguments in a text file in the experiment directory.
    with open(os.path.join(exp_dir, 'args.txt'), 'w') as f:
        print('Save config setting at', os.path.join(exp_dir, 'args.txt'))
        for arg in vars(config):
            f.write(arg + ' ' + str(getattr(config, arg)) + '\n')

    """Init model"""
    print('Loading model to GPU...')
    deepVO = DeepVO(config.img_w, config.img_h, config.seq_len, config.batch_size,
                    activation=config.activation,
                    parameterization=config.outputParameterization,
                    dropout=config.dropout,
                    flownet_weights_path=config.loadModel,
                    num_lstm_cells=config.num_lstm_cells)

    deepVO.init_weights()
    deepVO.cuda()
    print('Finish!')

    """Set hyper-parameters"""
    # Optimizer
    if config.optMethod == 'adam':
        optimizer = optim.Adam(deepVO.parameters(),
                               lr=config.lr,
                               betas=(config.beta1, config.beta2),
                               weight_decay=config.weight_decay,
                               amsgrad=False)
    elif config.optMethod == 'sgd':
        optimizer = optim.SGD(deepVO.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay,
                              nesterov=False)
    else:
        optimizer = optim.Adagrad(deepVO.parameters(), lr=config.lr, lr_decay=config.lr_decay,
                                  weight_decay=config.weight_decay)

    # Scheduler
    scheduler = None
    if config.lrScheduler is not None:
        if config.lrScheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        elif config.lrScheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    """Load dataset"""
    train_seq = [1]
    train_startFrames = [0]
    train_endFrames = [1100]
    val_seq = [1]
    val_startFrames = [0]
    val_endFrames = [1100]

    train_set = KITTIDataset(config.datadir,
                             sequences=train_seq,
                             sequence_len=config.seq_len,
                             start_frames=train_startFrames, end_frames=train_endFrames,
                             parameterization=config.outputParameterization,
                             width=config.img_w, height=config.img_h)
    val_set = KITTIDataset(config.datadir,
                           sequences=val_seq,
                           sequence_len=config.seq_len,
                           start_frames=val_startFrames, end_frames=val_endFrames,
                           parameterization=config.outputParameterization,
                           width=config.img_w, height=config.img_h)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=config.workers)

    """Main loop"""
    min_val_loss = None

    loss_each_epoch = []
    val_loss_each_epoch = []

    for epoch in range(config.epochs):
        # Train
        criterion = torch.nn.MSELoss(reduction='sum')
        print('===> Training: ' + str(epoch + 1) + '/' + str(config.epochs))
        avg_train_loss, avg_train_R_loss, avg_train_t_loss = train(train_loader, deepVO,
                                                                   criterion, optimizer, config, scheduler)
        loss_each_epoch.append(avg_train_loss)

        # Valid
        print('===> Validating: ' + str(epoch + 1) + '/' + str(config.epochs))
        avg_val_loss, avg_val_R_loss, avg_val_t_loss = val(val_loader, deepVO, criterion)
        val_loss_each_epoch.append(avg_val_loss)

        # Save checkpoint
        if min_val_loss is None:
            min_val_loss = avg_val_loss
        if avg_val_loss < min_val_loss:
            print('Saving best model...')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': deepVO.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, exp_dir, is_best=True)
            print('Finish!')
        if epoch != 0 and epoch % 5 == 0:
            print('Saving checkpoint...')
            save_checkpoint({
                'state_dict': deepVO.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, exp_dir, is_best=False, filename=f'checkpoint_{epoch}.pth.tar')
            print('Finish!')

    # Draw training, valid curves
    fig, ax = plt.subplots(1)
    ax.plot(range(config.epochs), loss_each_epoch, 'b', label='train')
    ax.plot(range(config.epochs), val_loss_each_epoch, 'r', label='total_val')
    ax.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    fig.savefig(os.path.join(exp_dir, 'loss_curve'))

    # Use the best weights to draw map
    print('===> Testing')
    test(exp_dir, val_set, deepVO)


def train(loader, model, criterion, optimizer, config, scheduler):
    model.train()  # Switch to train mode

    # Cache loss to tracking training progress
    sum_loss = 0
    R_sum_loss = 0
    t_sum_loss = 0

    stat_bar = tqdm(enumerate(loader), total=len(loader), position=0, leave=True)
    for idx, data in stat_bar:
        tensor, R, t, _ = data

        # Load all data to CUDA
        tensor = tensor.cuda(non_blocking=True)
        R = R.cuda(non_blocking=True)
        t = t.cuda(non_blocking=True)

        R_pred, t_pred = model.forward(tensor)

        R_pred = R_pred.permute(1, 0, 2)
        t_pred = t_pred.permute(1, 0, 2)

        R_loss = criterion(R_pred, R)
        t_loss = criterion(t_pred, t)

        loss = config.scf * R_loss + t_loss

        # float() is really important here.
        # It helps to avoid storing autograd history which may lead to out of memory.
        sum_loss += float(loss)
        R_sum_loss += float(R_loss)
        t_sum_loss += float(t_loss)

        optimizer.zero_grad()
        loss.backward()
        if config.gradClip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradClip)
        optimizer.step()
        if config.lrScheduler is not None:
            scheduler.step()

        stat_bar.set_description('Train {}/{}:'.format(idx, len(loader)))
        stat_bar.set_postfix({'r_loss': '{:.6f}'.format(R_sum_loss / (idx + 1)),
                              't_loss': '{:.6f}'.format(t_sum_loss / (idx + 1)),
                              'total_loss': '{:.6f}'.format(sum_loss / (idx + 1))
                              })

    avg_loss = sum_loss / len(loader)
    R_avg_loss = R_sum_loss / len(loader)
    t_avg_loss = t_sum_loss / len(loader)

    return avg_loss, R_avg_loss, t_avg_loss


def val(loader, model, criterion):
    model.eval()

    # Cache loss to tracking training progress
    sum_loss = 0
    R_sum_loss = 0
    t_sum_loss = 0

    stat_bar = tqdm(enumerate(loader), total=len(loader), position=0, leave=True)

    with torch.no_grad():
        for idx, data in stat_bar:
            tensor, R, t, _ = data

            # Load all data to CUDA
            tensor = tensor.cuda(non_blocking=True)
            R = R.cuda(non_blocking=True)
            t = t.cuda(non_blocking=True)

            R_pred, t_pred = model.forward(tensor)

            R_pred = R_pred.permute(1, 0, 2)
            t_pred = t_pred.permute(1, 0, 2)

            R_loss = criterion(R, R_pred)
            t_loss = criterion(t, t_pred)

            loss = config.scf * R_loss + t_loss

            sum_loss += loss.item()
            R_sum_loss += R_loss.item()
            t_sum_loss += t_loss.item()

            stat_bar.set_description('Validate {}/{}:'.format(idx, len(loader)))
            stat_bar.set_postfix({'r_loss': '{:.6f}'.format(R_sum_loss / (idx + 1)),
                                  't_loss': '{:.6f}'.format(t_sum_loss / (idx + 1)),
                                  'total_loss': '{:.6f}'.format(sum_loss / (idx + 1))
                                  })

    avg_loss = sum_loss / len(loader)
    R_avg_loss = R_sum_loss / len(loader)
    t_avg_loss = t_sum_loss / len(loader)

    return avg_loss, R_avg_loss, t_avg_loss


def test(exp_dir, dataset, model):
    best_path = os.path.join(exp_dir, 'model_best.pth.tar')
    print('Loading best weights./..')
    state_dict = torch.load(best_path)['state_dict']
    model.load_state_dict(state_dict)
    print('Finish')
    model.eval()

    stat_bar = tqdm(enumerate(dataset), total=len(dataset), position=0, leave=True)

    with torch.no_grad():
        for idx, data in stat_bar:
            tensor, _, _, vid_seq_id = data
            tensor = tensor.unsqueeze(0)
            vid_seq_id = vid_seq_id.item()
            path = os.path.join(exp_dir, 'plots', 'traj', str(vid_seq_id).zfill(2), 'traj.txt')

            # Load input to CUDA
            tensor = tensor.cuda(non_blocking=True)

            R_pred, t_pred = model.forward(tensor)

            R_pred = R_pred.permute(1, 0, 2)
            t_pred = t_pred.permute(1, 0, 2)

            write_pred_traj(path, R_pred, t_pred)

    vid_seq_ids = dataset.df.sequence_idx.unique()

    for vid_seq_id in vid_seq_ids:
        seqLen = len(dataset)
        traj_pred_file = os.path.join(exp_dir, 'plots', 'traj', str(vid_seq_id).zfill(2), 'traj.txt')
        if os.path.exists(traj_pred_file):
            traj_predicts = np.loadtxt(traj_pred_file)
            plot_seq(exp_dir, vid_seq_id, seqLen, traj_predicts, config.datadir, config)


if __name__ == '__main__':
    main()
