import os
import time
import argparse
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import video_transforms
import models
import datasets
from scripts.eval_ucf101_pytorch.utils import *
from tensorboardX import SummaryWriter

# spatial_log_path = 'train_spatial_log.txt'
# temporal_log_path = 'train_temporal_log.txt'
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--data', metavar='DIR',default='/Datasets/UCF101_jpegs_256/',
                    help='path to dataset')
parser.add_argument('--pretrained_model_path', metavar='MODEL_PATH',default=None,
                    help='loading params to the model')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=["rgb", "flow"],
                    help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "mini_kinetics"],
                    help='dataset: ucf101 | kinetics')
parser.add_argument('--optimizer', default='SGD',
                    choices=["SGD", "Adam"])
parser.add_argument('--dropout', default=0.7, type=float)                        
parser.add_argument('--arch', '-a', metavar='ARCH', default='convnext_tiny')
                    # choices=model_names,
                    # help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: rgb_vgg16)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100,200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
params = dict()
params['gpu'] = [4,5]

def main():
    global args, best_prec1
    args = parser.parse_args()
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    if args.modality == 'rgb':
        log = my_log('train_spatial_%s_log.txt'%args.dataset)
        logdir = './log/spatial_%s'%args.dataset
    else:
        log = my_log('train_temporal_%s_log.txt'%args.dataset)
        logdir = './log/temporal_%s'%args.dataset
    log_dir = os.path.join(logdir,cur_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log.write('参数如下：%s'%args)
    writer = SummaryWriter(log_dir=log_dir)
    # create model
    print("Building model ... ")
    if args.dataset == 'mini_kinetics':
        num_classes = 200
    elif args.dataset == 'ucf101':
        num_classes = 101
    else:
        print('wrong dataset!')
    model = build_model(args.pretrained_model_path, args.dropout, num_classes)
    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(params['gpu'][0])
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08) #weight_decay注意未设置
    else:
        raise AttributeError('only SGD and Adam are provided!')

    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))
    log.write("Saving everything to directory %s." % (args.resume))

    cudnn.benchmark = True

    # Data transforming
    if args.modality == "rgb":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.new_length
        clip_std = [0.229, 0.224, 0.225] * args.new_length
    elif args.modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.new_length
        clip_std = [0.226, 0.226] * args.new_length
    else:
        print("No such modality. Only rgb and flow supported.")

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.MultiScaleCrop((224, 224), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])

    # data loading
    if args.dataset == 'mini_kinetics':
        train_split_file = "/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/mini_kinetics_200_train.txt"
        val_split_file = "/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/mini_kinetics_200_val.txt"
    else:
        train_setting_file = "train_%s_split%d.txt" % (args.modality, args.split)
        train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
        val_setting_file = "val_%s_split%d.txt" % (args.modality, args.split)
        val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                    source=train_split_file,
                                                    phase="train",
                                                    modality=args.modality,
                                                    is_color=is_color,
                                                    new_length=args.new_length,
                                                    new_width=args.new_width,
                                                    new_height=args.new_height,
                                                    video_transform=train_transform)
    val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                  source=val_split_file,
                                                  phase="val",
                                                  modality=args.modality,
                                                  is_color=is_color,
                                                  new_length=args.new_length,
                                                  new_width=args.new_width,
                                                  new_height=args.new_height,
                                                  video_transform=val_transform)
    info = '{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset))
    print(info)
    log.write(info)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, log)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log, writer)

        # evaluate on validation set
        prec1 = 0.0
        if (epoch + 1) % args.save_freq == 0:
            prec1 = validate(val_loader, model, criterion, log, epoch, writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_name = "%s_epoch%03d_split%s.pth" % (args.arch, (epoch + 1), args.split)
            save_checkpoint({
                'split': args.split,
                'dataset': args.dataset,
                'model': args.arch,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),  #注意state_dict在这
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_name, args.resume)

def build_model(pretrained_model_path = None, dropout = 0.8 ,num_classes=200):
    if pretrained_model_path:
        model = models.__dict__[args.arch](pretrained=False, num_classes=num_classes, dropout=dropout)
        pretrained_dict = torch.load(pretrained_model_path)        #torch.save保存的字典有哪些，读取就有哪些。
        model.load_state_dict(pretrained_dict['state_dict'])     #给model.state_dict()加载更新值。
    else:
        model = models.__dict__[args.arch](pretrained=True, num_classes=num_classes, dropout=dropout)
    # model.cuda()
    return model

def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0
    acc3_mini_batch = 0.0

    for i, (input, target) in enumerate(train_loader):

        input = input.float().cuda(params['gpu'][0],non_blocking=True)
        target = target.cuda(params['gpu'][0],non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)


        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))  #取data而无梯度传播
        acc_mini_batch += prec1.item()
        acc3_mini_batch += prec3.item()
        loss = criterion(output, target_var)
        loss = loss / args.iter_size
        loss_mini_batch += loss.item()
        loss.backward()

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            # losses.update(loss_mini_batch/args.iter_size, input.size(0))
            # top1.update(acc_mini_batch/args.iter_size, input.size(0))
            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch/args.iter_size, input.size(0))
            top3.update(acc3_mini_batch/args.iter_size, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0
            acc3_mini_batch = 0

            if (i+1) % args.print_freq == 0:
                info ='Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1, top3=top3)
                log.write(info)
                print(info)
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top3_acc_epoch', top3.avg, epoch)        

@torch.no_grad()
def validate(val_loader, model, criterion, log, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.float().cuda(params['gpu'][0],non_blocking=True)
        target = target.cuda(params['gpu'][0],non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = 'Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1, top3=top3)
            print(info)
            log.write(info)
    info = ' 验证集精度为： Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3)
    print(info)
    log.write(info)
    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top3_acc_epoch', top3.avg, epoch)       
    return top1.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, '%s_model_best_%s.pth'%(state['model'],state['dataset']))
    torch.save(state, cur_path)    #save和load相对应。save不止可以存model.state_dict()这一个属性。此处就还存了其它键值。
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, log):

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    # decay = 0.1 ** (epoch//args.lr_steps)
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    log.write("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()