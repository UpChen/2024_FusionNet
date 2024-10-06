import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import joint_transforms
from config import VMD_training_root, VMD_test_root
from dataset.VShadow_crosspairwise_query_other import CrossPairwiseImg
from misc import AvgMeter, check_mkdir
# from networks.TVSD import TVSD
# from networks.VMD_network import VMD_Network
from torch.optim.lr_scheduler import StepLR
import math
from util.loss.losses import lovasz_hinge, binary_xloss
import random
import torch.nn.functional as F
import numpy as np
# from apex import amp
import time
import argparse
import importlib
from utils import backup_code
from utils import _sigmoid


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.deterministic = True
cudnn.benchmark = False

ckpt_path = './output'
# exp_name = 'VMD'

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='FusionNet', help='exp name')
parser.add_argument('--model', type=str, default='FusionNet', help='model name')
parser.add_argument('--gpu', type=str, default='0', help='used gpu id')
# parser.add_argument('--gpu', type=str, default='0,1', help='used gpu id')
parser.add_argument('--batchsize', type=int, default=5, help='train batch')
# parser.add_argument('--batchsize', type=int, default=8, help='train batch')
parser.add_argument('--bestonly', action="store_true", help='only best model')

cmd_args = parser.parse_args()
exp_name = cmd_args.exp
model_name = cmd_args.model
gpu_ids = cmd_args.gpu
train_batch_size = cmd_args.batchsize

VMD_file = importlib.import_module('networks.' + model_name)
VMD_Network = VMD_file.FusionNet
# print(torch.__version__)


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
# print(torch.cuda.device_count())

args = {
    # 'exp_name': exp_name,
    'change': 'pytorch 1.9 cuda11, FusionNet',
    'max_epoch': 15,
    # 'train_batch_size': 10,
    'last_iter': 0,
    'finetune_lr': 1e-5,
    # 'finetune_lr': 6e-5,
    # 'scratch_lr': 6e-4,
    'scratch_lr': 1e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    # 'scale': 384,
    'scale': 512,
    'multi-scale': None,
    # 'gpu': '4,5',
    # 'multi-GPUs': True,
    'fp16': False,
    'warm_up_epochs': 3,
    'seed': 1234
}

# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

# multi-GPUs training
if len(gpu_ids.split(',')) > 1:
    # print(1)
    batch_size = train_batch_size * len(gpu_ids.split(','))
    # print("batch_size" + str(batch_size))
# single-GPU training
else:
    torch.cuda.set_device(0)
    batch_size = train_batch_size

joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale'])),
    joint_transforms.RandomHorizontallyFlip()
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

print('=====>Dataset loading<======')
training_root = [VMD_training_root] # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
train_set = CrossPairwiseImg(training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=batch_size,  drop_last=True, num_workers=8, shuffle=True)

val_set = CrossPairwiseImg([VMD_test_root], val_joint_transform, img_transform, target_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False)

print("max epoch:{}".format(args['max_epoch']))

# ce_loss = nn.CrossEntropyLoss()
segmentation_loss = binary_xloss
lovasz_hinge = lovasz_hinge

exp_time = datetime.datetime.now()
log_dir_path = os.path.join(ckpt_path, exp_name, str(exp_time))
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
check_mkdir(log_dir_path)
log_path = os.path.join(ckpt_path, exp_name, str(exp_time), 'train_log.txt')
val_log_path = os.path.join(ckpt_path, exp_name, str(exp_time), 'val_log.txt')
train_writer = SummaryWriter(log_dir=os.path.join(log_dir_path, 'train'), comment='train')

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def loss_calc3(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if len(gpu_ids.split(',')) > 1:
        net = torch.nn.DataParallel(VMD_Network()).cuda().train()
        # for name, param in net.named_parameters():
        #     if 'backbone' in name:
        #         print(name)
        # net = net.apply(freeze_bn) # freeze BN
        params = [
            {"params": net.module.segformer.parameters(), "lr": args['finetune_lr']},
            # {"params": net.module.decode_head.parameters(), "lr": args['finetune_lr']},
            {"params": net.module.aspp.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.ra_attention_spatial_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.ra_attention_spatial_low.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.ra_attention_low.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.ra_attention_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.project.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.final_pre.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.conv1.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.conv2.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.bn1.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.bn2.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.globalAvgPool.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.fc1.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.fc2.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.fc3.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.fc4.parameters(), "lr": args['scratch_lr']},
            {"params": net.module.final_query.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.final_other.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.ra_attention_4.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.encoder.backbone.parameters(), "lr": args['finetune_lr']},
            # {"params": net.module.encoder.aspp.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.encoder.final_pre.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.ra_attention.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.project.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.final_pre.parameters(), "lr": args['scratch_lr']}
        ]
    # single-GPU training
    else:
        net = VMD_Network().cuda().train()
        # for name, param in net.named_parameters():
        #     print(name)
        # net = net.apply(freeze_bn) # freeze BN
        params = [
            {"params": net.segformer.parameters(), "lr": args['finetune_lr']},
            # {"params": net.decode_head.parameters(), "lr": args['finetune_lr']},
            {"params": net.aspp.parameters(), "lr": args['scratch_lr']},
            {"params": net.ra_attention_spatial_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.ra_attention_spatial_low.parameters(), "lr": args['scratch_lr']},
            # {"params": net.fc.parameters(), "lr": args['scratch_lr']},
            {"params": net.ra_attention_low.parameters(), "lr": args['scratch_lr']},
            {"params": net.ra_attention_high.parameters(), "lr": args['scratch_lr']},
            {"params": net.project.parameters(), "lr": args['scratch_lr']},
            {"params": net.final_pre.parameters(), "lr": args['scratch_lr']},
            # {"params": net.conv1.parameters(), "lr": args['scratch_lr']},
            # {"params": net.conv2.parameters(), "lr": args['scratch_lr']},
            # {"params": net.bn1.parameters(), "lr": args['scratch_lr']},
            # {"params": net.bn2.parameters(), "lr": args['scratch_lr']},
            # {"params": net.globalAvgPool.parameters(), "lr": args['scratch_lr']},
            # {"params": net.fc1.parameters(), "lr": args['scratch_lr']},
            # {"params": net.fc2.parameters(), "lr": args['scratch_lr']},
            # {"params": net.fc3.parameters(), "lr": args['scratch_lr']},
            {"params": net.ffm.parameters(), "lr": args['scratch_lr']},
            {"params": net.final_query.parameters(), "lr": args['scratch_lr']},
            # {"params": net.final_other.parameters(), "lr": args['scratch_lr']},
            # {"params": self.backbone.deformation_trajectory_attention_1.parameters(), "lr": self.scratch_learning_rate},
            # {"params": self.backbone.deformation_trajectory_attention_2.parameters(), "lr": self.scratch_learning_rate},
            # {"params": self.backbone.deformation_trajectory_attention_3.parameters(), "lr": self.scratch_learning_rate},
            # {"params": self.projection.parameters(), "lr": self.scratch_learning_rate},
        ]

    # optimizer = optim.SGD(params, momentum=args['momentum'], weight_decay=args['weight_decay'])
    optimizer = optim.AdamW(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    # warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] else 0.5 * \
    #                          (math.cos((epoch-args['warm_up_epochs'])/(args['max_epoch']-args['warm_up_epochs'])*math.pi)+1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # change learning rate after 20000 iters

    # check_mkdir(ckpt_path)
    # check_mkdir(os.path.join(ckpt_path, exp_name))
    # backup_code(".", os.path.join(ckpt_path, exp_name, "backup_code"))
    open(log_path, 'w').write(str(args) + '\n\n')
    if args['fp16']:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    train(net, optimizer)


def train(net, optimizer):
    curr_epoch = 1
    curr_iter = 1
    start = 0
    best_mae = 100.0

    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7, loss_record8 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record9, loss_record10, loss_record11 = AvgMeter(), AvgMeter(), AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))
        # train_iterator = tqdm(train_loader, desc=f'Epoch: {curr_epoch}', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|')
        # tqdm(train_loader, total=len(train_loader))
        train_writer.add_scalar('lr/parameter_group[0]', optimizer.param_groups[0]['lr'], curr_epoch)
        train_writer.add_scalar('lr/parameter_group[1]', optimizer.param_groups[1]['lr'], curr_epoch)
        train_writer.add_scalar('lr/parameter_group[2]', optimizer.param_groups[2]['lr'], curr_epoch)
        # train_writer.add_scalar('lr/parameter_group[3]', optimizer.param_groups[3]['lr'], curr_epoch)
        # train_writer.add_scalar('lr/parameter_group[4]', optimizer.param_groups[4]['lr'], curr_epoch)

        for i, sample in enumerate(train_iterator):

            exemplar, exemplar_gt = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda()
            query, query_gt = sample['query'].cuda(), sample['query_gt'].cuda()
            other, other_gt = sample['other'].cuda(), sample['other_gt'].cuda()
            # exemplar_guass = sample['exemplar_guass'].cuda()
            # query_guass = sample['query_guass'].cuda()

            optimizer.zero_grad()

            exemplar_pre, query_pre1, query_pre2, query_final, other_final = net(exemplar, query, other)
            # heat = _sigmoid(heat)
            # exemplar_logits = exemplar_outputs.logits
            # query_logits = query_outputs.logits
            # print(logits.shape)
            # exemplar_pre = torch.nn.functional.interpolate(
            #     exemplar_logits, size=exemplar.size()[2:], mode="bilinear", align_corners=False
            # )
            # query_pre = torch.nn.functional.interpolate(
            #     query_logits, size=query.size()[2:], mode="bilinear", align_corners=False
            # )
            # print(exemplar_pre.shape)
            exemplar_bce_loss = segmentation_loss(exemplar_pre, query_gt)
            # query_bce_loss = segmentation_loss(query_pre, query_gt)
            query_pre1_bce_loss = segmentation_loss(query_pre1, query_gt)
            query_pre2_bce_loss = segmentation_loss(query_pre2, query_gt)
            query_final_bce_loss = segmentation_loss(query_final, query_gt)
            other_bce_loss = segmentation_loss(other_final, other_gt)
            # bce_loss2 = binary_xloss(query_pre, query_gt)
            # bce_loss3 = binary_xloss(other_pre, other_gt)

            exemplar_hinge_loss = lovasz_hinge(exemplar_pre, query_gt)
            # query_hinge_loss = lovasz_hinge(query_pre, query_gt)
            query_pre1_hinge_loss = lovasz_hinge(query_pre1, query_gt)
            query_pre2_hinge_loss = lovasz_hinge(query_pre2, query_gt)
            query_final_hinge_loss = lovasz_hinge(query_final, query_gt)
            # query_hinge_loss = lovasz_hinge(query_pre, query_gt)
            other_hinge_loss = lovasz_hinge(other_final, other_gt)
            # loss_hinge2 = lovasz_hinge(query_pre, query_gt)
            # loss_hinge3 = lovasz_hinge(other_pre, other_gt)
            # loss_heat = loss_calc3(heat, query_guass)

            # loss_hinge_examplar = lovasz_hinge(examplar_final, exemplar_gt)
            # loss_hinge_query = lovasz_hinge(query_final, query_gt)
            # loss_hinge_other = lovasz_hinge(other_final, other_gt)
            
            loss_seg = (exemplar_bce_loss + query_pre1_bce_loss + query_pre2_bce_loss + query_final_bce_loss +
                        other_bce_loss + exemplar_hinge_loss + query_pre1_hinge_loss + query_pre2_hinge_loss +
                        query_final_hinge_loss + other_hinge_loss)
            
            # loss_seg = bce_loss1 + bce_loss2 + bce_loss3 + loss_hinge1 + loss_hinge2 + loss_hinge3
            # classification loss
            
            # scene_labels = torch.zeros(scene_logits.shape[0], dtype=torch.long).cuda()
            # cla_loss = ce_loss(scene_logits, scene_labels) * 10
            # loss = loss_seg + cla_loss
            loss = loss_seg

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 12)  # gradient clip
            optimizer.step()  # change gradient

            loss_record1.update(loss.item(), batch_size)
            loss_record2.update(exemplar_bce_loss.item(), batch_size)
            loss_record3.update(query_pre1_bce_loss.item(), batch_size)
            loss_record4.update(query_pre2_bce_loss.item(), batch_size)
            loss_record5.update(query_final_bce_loss.item(), batch_size)
            loss_record6.update(other_bce_loss.item(), batch_size)
            loss_record7.update(exemplar_hinge_loss.item(), batch_size)
            loss_record8.update(query_pre1_hinge_loss.item(), batch_size)
            loss_record9.update(query_pre2_hinge_loss.item(), batch_size)
            loss_record10.update(query_final_hinge_loss.item(), batch_size)
            loss_record11.update(other_hinge_loss.item(), batch_size)

            train_writer.add_scalar('loss/total_loss', loss_record1.avg, curr_iter)
            train_writer.add_scalar('loss/exemplar_bce_loss', loss_record2.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre1_bce_loss', loss_record3.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre2_bce_loss', loss_record4.avg, curr_iter)
            train_writer.add_scalar('loss/query_final_bce_loss', loss_record5.avg, curr_iter)
            train_writer.add_scalar('loss/other_bce_loss', loss_record6.avg, curr_iter)
            train_writer.add_scalar('loss/exemplar_hinge_loss', loss_record7.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre1_hinge_loss', loss_record8.avg, curr_iter)
            train_writer.add_scalar('loss/query_pre2_hinge_loss', loss_record9.avg, curr_iter)
            train_writer.add_scalar('loss/query_final_hinge_loss', loss_record10.avg, curr_iter)
            train_writer.add_scalar('loss/other_hinge_loss', loss_record11.avg, curr_iter)
            # train_writer.add_scalar('loss/loss_heat', loss_record6.avg, curr_iter)

            logged_images = {
                "images/clip_1": (reverse_normalize(exemplar[0]).cpu().numpy() * 255).astype(np.uint8),
                "images/clip_2": (reverse_normalize(query[0]).cpu().numpy() * 255).astype(np.uint8),
                # "images/clip_3": (self.reverse_normalize(exemplar[2]).cpu().numpy() * 255).astype(np.uint8),
                "preds/clip_1": (exemplar_pre[0] > 0.5).to(torch.int8),
                "preds/clip_2": (query_final[0] > 0.5).to(torch.int8),
                # "preds/clip_3": (final_outputs[2] > 0.5).to(torch.int8),
                "labels/clip_1": exemplar_gt[0].unsqueeze(0) if len(exemplar_gt[0].size()) == 2 else exemplar_gt[0],
                "labels/clip_2": query_gt[0].unsqueeze(0) if len(query_gt[0].size()) == 2 else query_gt[0],
                # "labels/clip_3": labels[2].unsqueeze(0) if len(labels[2].size()) == 2 else labels[2],
            }

            for image_name, image in logged_images.items():
                # print(image.shape)
                train_writer.add_image("{}".format(image_name), image, curr_iter)  # self.current_epoch)

            curr_iter += 1

            log = ("epochs:%d, iter: %d, loss: %f5, exemplar_bce_loss: %f5, query_pre1_bce_loss: %f5, query_pre2_bce_loss: %f5,  "
                   "query_final_bce_loss: %f5, other_bce_loss: %f5, exemplar_hinge_loss:%f5, query_pre1_hinge_loss:%f5, "
                   "query_pre2_hinge_loss:%f5, query_final_hinge_loss: %f5, other_hinge_loss:%f5, lr: %f8")%\
                  (curr_epoch, curr_iter, loss_record1.avg, loss_record2.avg,  loss_record3.avg, loss_record4.avg,
                   loss_record5.avg, loss_record6.avg, loss_record7.avg, loss_record8.avg, loss_record9.avg,
                   loss_record10.avg, loss_record11.avg, optimizer.param_groups[0]['lr'])

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.perf_counter() - start)
                start = time.perf_counter()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
                # train_iterator.set_description(log_time)
            open(log_path, 'a').write(log + '\n')

        if curr_epoch % 1 == 0 and not cmd_args.bestonly:
            # if args['multi-GPUs']:
            if len(gpu_ids.split(',')) > 1:
                # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                if args['fp16']:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, str(exp_time), f'{curr_epoch}.pth'))
            else:
                # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                if args['fp16']:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, str(exp_time), f'{curr_epoch}.pth'))


        current_mae = val(net, curr_epoch)

        net.train() # val -> train
        if current_mae < best_mae:
            best_mae = current_mae
            if len(gpu_ids.split(',')) > 1:
                if args['fp16']:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            else:
                if args['fp16']:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            torch.save(checkpoint, os.path.join(ckpt_path, exp_name, str(exp_time), 'best_mae.pth'))



        if curr_epoch > args['max_epoch']:
            # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            return

        curr_epoch += 1
        # scheduler.step()  # change learning rate after epoch


def val(net, epoch):
    mae_record = AvgMeter()
    net.eval()
    with torch.no_grad():
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            query, other = sample['query'].cuda(), sample['other'].cuda()
            query_gt, other_gt = sample['query_gt'].cuda(), sample['other_gt'].cuda()
            exemplar, exemplar_gt = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda()
            # exemplar_guass = sample['exemplar_guass'].cuda()

            _, _, _, query_final, _, = net(exemplar, query, other)
            # logits = outputs.logits
            # exemplar_pre = torch.nn.functional.interpolate(
            #     logits, size=exemplar.size()[2:], mode="bilinear", align_corners=False
            # )

            res = (query_final.data > 0).to(torch.float32).squeeze(0)
                        # res = torch.sigmoid(exemplar_pre.squeeze())
            mae = torch.mean(torch.abs(res - query_gt.squeeze(0)))

            batch_size = query.size(0)
            mae_record.update(mae.item(), batch_size)
            # prediction = np.array(transforms.Resize((h, w))(to_pil(res.cpu())))
        train_writer.add_scalar("val/mae", mae_record.avg, epoch)
        log = "val: iter: %d, mae: %f5" % (epoch, mae_record.avg)
        print(log)
        open(val_log_path, 'a').write(log + '\n')
        return mae_record.avg


def reverse_normalize(normalized_image):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    inv_tensor = inv_normalize(normalized_image)
    return inv_tensor


if __name__ == '__main__':
    main()
    # a = torch.rand(2, 2)
    # print(a)
    # b = torch.ones(2, 2)
    # print(b)
    # print(a*b)
    # print(torch.matmul(a, b))
    #
    # str1 = str("xumingchen-qiuxiaoyu")
    # str2 = str("xumingchen")
    # str3 = str("qiuxiaoyu")
    # str4 = str("mingchen-qiuxiaoyu")
    # str5 = str("xumingchen-qiuxiaoyu1")
    # str6 = str("xumingchen-qiuxiao2")
    # str7 = str("xumingchen-xiaoyu3")
    # str8 = str("chen-qiuxiaoyu4")
    # # if "xumingchen" and "qiuxiaoyu" in str3:
    # #     print(True)
    #
    # if str3.find("xumingchen") != -1 and str3.find("qiuxiaoyu") != -1:
    #     print(True)