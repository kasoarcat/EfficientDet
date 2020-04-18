
import os
import platform

IMAGE_SIZE = 512
NUM_EPOCH = 40
BATCH_SIZE = 4
WORKS = 2
LEARNING_RATE = 1e-4

# NETWORK = 'efficientdet-d0'
# NETWORK = 'efficientdet-d1'
# NETWORK = 'efficientdet-d2'
# NETWORK = 'efficientdet-d3'
NETWORK = 'efficientdet-d4'
# NETWORK = 'efficientdet-d5'
# NETWORK = 'efficientdet-d6'
# NETWORK = 'efficientdet-d7'

if platform.system() == 'Linux':
    import subprocess
    import sys
    import shutil

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    install('install/pycocotools-2.0-cp36-cp36m-linux_x86_64.whl')
    install('install/pytoan-0.6.4-py3-none-any.whl')
    install('install/imgaug-0.2.6-py3-none-any.whl')
    install('install/albumentations-0.4.5-py3-none-any.whl')

from tqdm import tqdm
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.efficientdet import EfficientDet
from models.losses import FocalLoss
from datasets import VOCDetection, H5CoCoDataset, CocoDataset, get_augumentation, detection_collate, Resizer, Normalizer, Augmenter, collater
from utils import EFFICIENTDET, get_state_dict
from eval import evaluate, evaluate_coco
import json

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='show', choices=['limit', 'h5', 'show'], type=str, help='limit, h5, show')
parser.add_argument('--dataset_root', default='/mnt/marathon', help='Dataset root directory path')
parser.add_argument('--network', default=NETWORK, type=str, help='efficientdet-[d0, d1, ..]')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_epoch', default=NUM_EPOCH, type=int, help='Num epoch for training')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Batch size for training')
parser.add_argument('--num_class', default=3, type=int, help='Number of class used in model')
parser.add_argument('--limit', help='limit', type=int, nargs=2, default=(0, 0))
parser.add_argument('--device', default=[0], type=list, help='Use CUDA to train model')
parser.add_argument('--grad_accumulation_steps', default=1, type=int, help='Number of gradient accumulation steps')
parser.add_argument('--lr', '--learning-rate', default=LEARNING_RATE, type=float, help='initial learning rate')
parser.add_argument('--image_size', help='image size', type=int, default=IMAGE_SIZE)
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./saved/weights/', type=str, help='Directory for saving checkpoint models')
parser.add_argument('-j', '--workers', default=WORKS, type=int, metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')

iteration = 1


def train(train_loader, model, scheduler, optimizer, epoch, args, epoch_loss_file, iteration_loss_file, steps_pre_epoch):
    global iteration
    # print("{} epoch: \t start training....".format(epoch))
    start = time.time()
    total_loss = []
    model.train()
    model.module.is_training = True
    model.module.freeze_bn()
    optimizer.zero_grad()
    for idx, (images, annotations) in enumerate(train_loader):
        # print('images.shape:', images.shape)
        images = images.cuda().float()
        annotations = annotations.cuda()
        classification_loss, regression_loss = model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            print('loss equal zero(0)')
            continue
        loss.backward()
        if (idx + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        total_loss.append(loss.item())
        mean_total_loss = np.mean(total_loss)
        print('\repoch: {} | iteration: {} | cls_loss: {:1.5f} | reg_loss: {:1.5f} | mean_loss: {:1.5f}'.format(epoch+1, iteration,
            classification_loss.item(), regression_loss.item(), mean_total_loss), end=' ' * 50)
        iteration_loss_file.write('{},{},{:1.5f},{:1.5f},{:1.5f}\n'.format(epoch+1, epoch * steps_pre_epoch + (iteration+1),
            float(classification_loss), float(regression_loss), mean_total_loss))    
        iteration_loss_file.flush()
        iteration += 1
    scheduler.step(np.mean(total_loss))
    mean_total_loss = np.mean(total_loss)
    print('time: {:.0f}'.format(time.time() - start))
    epoch_loss_file.write('{},{:1.5f}\n'.format(epoch+1, mean_total_loss))
    epoch_loss_file.flush()


def test(dataset, model, epoch, args, coco_eval_file):
    # print("{} epoch: \t start validation....".format(epoch+1))
    model = model.module
    model.eval()
    model.is_training = False
    with torch.no_grad():
        evaluate_coco(dataset, model, args.dataset, epoch, coco_eval_file)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            # args.rank = int(os.environ["RANK"])
            args.rank = 1
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    # args.num_class = train_dataset.num_classes()

    print('dataset:', args.dataset)
    print('network:', args.network)
    print('num_epoch:', args.num_epoch)
    print('batch_size:', args.batch_size)
    print('lr:', args.lr)
    print('image_size:', args.image_size)
    print('workers:', args.workers)
    print('num_class:', args.num_class)
    print('save_folder:', args.save_folder)
    print('limit:', args.limit)

    if args.dataset == 'h5':
        train_dataset = H5CoCoDataset('{}/train_small.hdf5'.format(args.dataset_root), 'train_small')
        valid_dataset = H5CoCoDataset('{}/test.hdf5'.format(args.dataset_root), 'test')
    else:
        train_dataset = CocoDataset(args.dataset_root, set_name='train_small',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(args.image_size)]),
                                # transform=get_augumentation('train'),
                                limit_len=args.limit[0])
        valid_dataset = CocoDataset(args.dataset_root, set_name='test',
                              transform=transforms.Compose([Normalizer(), Resizer(args.image_size)]), limit_len=args.limit[1])

    print('train_dataset:', len(train_dataset))
    print('valid_dataset:', len(valid_dataset))

    steps_pre_epoch = len(train_dataset) // args.batch_size
    print('steps_pre_epoch:', steps_pre_epoch)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)

    checkpoint = []
    if(args.resume is not None):
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        args.start_epoch = checkpoint['epoch'] + 1
        del params

    model = EfficientDet(num_classes=args.num_class,
                         network=args.network,
                         W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                         D_class=EFFICIENTDET[args.network]['D_class']
                         )
    
    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])
    
    del checkpoint

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            print('Run with DistributedDataParallel with divice_ids....')
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Run with DistributedDataParallel without device_ids....')
    elif args.gpu is not None:
        # print('using gpu:', args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cpu()
        # print('Run with DataParallel ....')
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) , optimizer, scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose=True)
    cudnn.benchmark = True
    
    iteration_loss_path = 'iteration_loss.csv'
    if os.path.isfile(iteration_loss_path):
        os.remove(iteration_loss_path)
    
    epoch_loss_path = 'epoch_loss.csv'
    if os.path.isfile(epoch_loss_path):
        os.remove(epoch_loss_path)
    
    eval_result_path = 'eval_result.csv'
    if os.path.isfile(eval_result_path):
        os.remove(eval_result_path)

    USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
    if USE_KAGGLE:
        iteration_loss_path = '/kaggle/working/' + iteration_loss_path
        epoch_loss_path = '/kaggle/working/' + epoch_loss_path
        eval_result_path = '/kaggle/working/' + eval_result_path
    
    with open(epoch_loss_path, 'a+') as epoch_loss_file, open(iteration_loss_path, 'a+') as iteration_loss_file, \
        open(eval_result_path, 'a+') as coco_eval_file:

        epoch_loss_file.write('epoch_num,mean_epoch_loss\n')
        iteration_loss_file.write('epoch_num,iteration,classification_loss,regression_loss,iteration_loss\n')
        coco_eval_file.write('epoch_num,map50\n')
        for epoch in range(args.start_epoch, args.num_epoch):
            train(train_loader, model, scheduler, optimizer, epoch, args, epoch_loss_file, iteration_loss_file, steps_pre_epoch)
            test(valid_dataset, model, epoch, args, coco_eval_file)


def main():
    args = parser.parse_args()
    USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
    if USE_KAGGLE:
        args.save_folder = '/kaggle/working/' + args.save_folder
    if(not os.path.exists(os.path.join(args.save_folder, args.dataset, args.network))):
        os.makedirs(os.path.join(args.save_folder, args.dataset, args.network))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '2'
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('device count:', ngpus_per_node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    return args


def write_result_csv(args):
    test_file = '{}/test/test.json'.format(args.dataset_root)
    with open(test_file) as f:
        data = json.load(f)
        image_dict = {}
        for image in data['images']:
            image_dict[image['id']] = image['file_name']

    USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
    result_json = 'test_bbox_results.json'
    result_csv = 'result.csv'
    if USE_KAGGLE:
        result_json = '/kaggle/working/' + result_json
        result_csv  = '/kaggle/working/' + result_csv
    with open(result_json) as f_result, open(result_csv, 'w') as f_out:
        anno_list = json.load(f_result)
        f_out.write("image_filename,label_id,x,y,w,h,confidence\n")
        for anno in anno_list:
            f_out.write("%s" % image_dict[anno['image_id']])
            x,y,w,h = anno['bbox']
            f_out.write(",{},{},{},{},{},{:.02f}".format(anno['category_id'], x,y,w,h, anno['score']))
            f_out.write('\n')


if __name__ == "__main__":
    args = main()
    write_result_csv(args)