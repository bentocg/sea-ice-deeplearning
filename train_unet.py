import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval import eval_net
from utils.data_handling import BasicDataset
from utils.models import UNet
from utils.data_handling import get_training_augmentation, get_validation_augmentation

dir_img_train = 'training_set/x/'
dir_mask_train = 'training_set/y/'
dir_checkpoint = 'checkpoints'
labels_file = 'training_set/classes.csv'


def train_net(net,
              device,
              epochs=5,
              thresh=0.8,
              pos_weight=50.0,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_size=256):
    train = BasicDataset(dir_img_train, dir_mask_train, labels_file, dataset='training',
                         augmentation=get_training_augmentation(img_size))
    val = BasicDataset(dir_img_train, dir_mask_train, labels_file, dataset='validation',
                       augmentation=get_validation_augmentation(img_size))
    n_train, n_val = len(train), len(val)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SIZE_{img_size}_WEIGHT_{pos_weight}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Positive weight: {pos_weight}
        Threshold:       {thresh}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:  {img_size}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=25)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).cuda())
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // batch_size) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    pred = torch.sigmoid(masks_pred)
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', 1 - imgs, global_step)

                    writer.add_images('masks/true', true_masks, global_step)
                    for thresh in range(5, 10):
                        writer.add_images(f'masks/pred_{thresh * 10}%', (pred > (thresh/10)).float(), global_step)


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=256,
                        help='Final size of images')
    parser.add_argument('-w', '--pos_weight', dest='weight', type=int, default=5,
                        help='weight for positive samples')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=1)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    net = torch.nn.DataParallel(net)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  pos_weight=args.weight,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
