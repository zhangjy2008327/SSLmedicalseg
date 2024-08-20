import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
#预训练100个epoch  微调是1个epoch
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from glob import glob
from torch.nn import init
# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose, OneOf
# import albumentations as albu

from RegSeg_8 import RegSeg
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import albumentations as A

from myloader import Dataset
from eval import evalResult
from loss_meter import bce_dice_loss
from io import BytesIO
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def My_Transformer():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.25),
            A.ShiftScaleRotate(shift_limit=0,p=0.25),
            A.CoarseDropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
        is_check_shapes=False
    )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--dataset', default='brats_2018', help='dataset name')

    parser.add_argument('--fpath', default='./dataset')

    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')

    parser.add_argument('--ngpus_per_node', default=2, type=int)

    parser.add_argument('--gpu', default='0', type=int, help='GPU id to use.')

    parser.add_argument('--img_ext', default='.png', type=str, help='type of the image')

    parser.add_argument('--mask_ext', default='.png', type=str, help='type of the mask')

    parser.add_argument('--img_dir', default='./dataset/brats_2018/fine-tuning', type=str, help='path of images')

    parser.add_argument('--log_dir', default='./log', type=str, help='path of log')

    parser.add_argument('--mask_dir', default='./dataset/brats_2018/fine-tuning_mask', type=str, help='path of masks')

    parser.add_argument('--test_dir', default='./dataset/brats_2018/test', type=str, help='path of test')

    parser.add_argument('--testmask_dir', default='./dataset/brats_2018/testmask', type=str, help='path of testmask')

    # parser.add_argument('--save_path', default='./best_model', type=str, help='path of best_model')

    parser.add_argument('--num_classes', default=1, type=int)

    parser.add_argument('--input_h', default=256, type=int)

    parser.add_argument('--input_w', default=256, type=int)

    config = parser.parse_args()

    return config
def save_checkpoint(path, model, key="model"):
    # save model state dict
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("checkpoint saved at", path)

def main():



    args = parse_args()  # parser.parse_arge()函数是用来解析命令行参数的
    args.ngpus_per_node = torch.cuda.device_count()  # 获取当前节点gpu可用数量



    img_ids = glob(f'{args.fpath}/{args.dataset}/fine-tuning/*{args.img_ext}')

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    test_ids = glob(f'{args.fpath}/{args.dataset}/test/*{args.img_ext}')

    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_ids]


    test_dataset = Dataset(
        img_ids=test_ids,
        img_dir=args.test_dir,
        mask_dir=args.testmask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=args.num_classes,
        transform=My_Transformer())

    # dataset = Dataset(
    #     img_ids=img_ids,
    #     img_dir=args.img_dir,
    #     mask_dir=args.mask_dir,
    #     img_ext=args.img_ext,
    #     mask_ext=args.mask_ext,
    #     num_classes=args.num_classes,
    #     transform=My_Transformer())

    # 定义交叉验证分割器
    #KDSB18和BUSI是5折    ISIC18和BraTS18是10折
    kfold = KFold(n_splits=10, shuffle=True)
    flag=0
    # 进行10折交叉验证训练和评估
    for fold, (train_indices, val_indices) in enumerate(kfold.split(img_ids)):

        train_ids = [img_ids[i] for i in train_indices]
        val_ids = [img_ids[i] for i in val_indices]


        # 创建训练集和验证集的数据加载器
        train_dataset = Dataset(
            img_ids=train_ids,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            img_ext=args.img_ext,
            mask_ext=args.mask_ext,
            num_classes=args.num_classes,
            transform=My_Transformer())

        val_dataset = Dataset(
            img_ids=val_ids,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            img_ext=args.img_ext,
            mask_ext=args.mask_ext,
            num_classes=args.num_classes,
            transform=My_Transformer())
        # test_dataset =
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = RegSeg()
        encoder_weights_path = './checkpoint/RegSeg_8_brats18_encoder.pth'
        model.encoder.load_state_dict(torch.load(encoder_weights_path), strict=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0001)

        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        
        model.to("cuda")
        model.train()
        best_val_metric = 0
        best_epoch = 0
        train_logs = {'epoch':[], 'loss':[], 'acc':[],'pre':[], 'rec':[], 'Miou':[], 'dice':[], 'HD':[], 'm_b_iou':[], 'MAE':[] }
        val_logs = {'epoch': [], 'acc': [], 'pre': [], 'rec': [], 'Miou': [], 'dice': [], 'HD': [], 'm_b_iou': [], 'MAE': []}
        test_logs = {'fold': [], 'acc': [], 'pre': [], 'rec': [], 'Miou': [], 'dice': [], 'HD': [], 'm_b_iou': [], 'MAE': []}
        torch.cuda.empty_cache()




        for epoch in range(args.epochs):  # 假设每个折进行10轮迭代
            training_loss = 0.0
            training_acc = 0
            training_pre = 0
            training_rec = 0
            training_miou = 0
            training_dice = 0
            training_mbiou = 0
            training_hd = 0
            training_mae = 0
            training_num = 0
            for (images, masks) in train_loader:

                image_t = images.to("cuda")
                mask_t = masks.to("cuda")

                outputs = model(image_t)

                masks_n = mask_t.clone()
                outputs_n = outputs

                loss = bce_dice_loss(masks_n, outputs_n)
                training_loss += loss.item()
                training_num +=1
                # print('------loss\n', loss.item())
                # print('zzz')
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                

                #zhibiao
                outputs_n = outputs_n.cpu()
                masks_n = masks_n.cpu()
                predicted = torch.sigmoid(outputs_n)

                # predicted = (predicted > 0.5).astype(np.float32)
                # masks = (masks > 0.5).astype(np.float32)
                outputs_n = np.where(outputs_n > 0.5, 1, 0)
                masks_n = np.where(masks_n > 0.5, 1, 0)

                acc, pre, rec, miou, dice, hd, mbiou, mae =  evalResult(masks_n, outputs_n)
                training_acc += acc
                training_pre += pre
                training_rec += rec
                training_miou += miou
                training_dice += dice
                training_hd += hd
                training_mae += mae
                training_mbiou += mbiou
                

                # print('xxx')
            # scheduler.step()
            training_loss = training_loss / training_num
            training_acc = training_acc / training_num
            training_pre = training_pre / training_num
            training_rec = training_rec / training_num
            training_miou = training_miou / training_num
            training_dice = training_dice / training_num
            training_hd = training_hd / training_num
            training_mae = training_mae / training_num
            training_mbiou = training_mbiou / training_num

            train_logs['epoch'].append(epoch+1)
            train_logs['loss'].append(training_loss)
            train_logs['acc'].append(training_acc)
            train_logs['pre'].append(training_pre)
            train_logs['rec'].append(training_rec)
            train_logs['Miou'].append(training_miou)
            train_logs['dice'].append(training_dice)
            train_logs['HD'].append(training_hd)
            train_logs['m_b_iou'].append(training_mbiou)
            train_logs['MAE'].append(training_mae)



            print('epoch--',epoch,'---loss---',training_loss)
        # print("ppp")
        # 在验证集上评估模型

            model.to("cuda")
            val_acc = 0
            val_pre = 0
            val_rec = 0
            val_miou = 0
            val_dice = 0
            val_hd = 0
            val_mbiou = 0
            val_mae = 0
            val_num = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to("cuda")
                    masks = masks.to("cuda")

                    outputs = model(images)
                    predicted = outputs

                    predicted = predicted.cpu()
                    masks = masks.cpu()
                    predicted = torch.sigmoid(predicted)

                    # predicted = (predicted > 0.5).astype(np.float32)
                    # masks = (masks > 0.5).astype(np.float32)
                    predicted = np.where(predicted > 0.5, 1, 0)
                    masks = np.where(masks > 0.5, 1, 0)


                    acc, pre, rec, miou, dice, hd, mbiou, mae = evalResult(masks, predicted)
                    val_acc += acc
                    val_pre += pre
                    val_rec += rec
                    val_miou += miou
                    val_dice += dice
                    val_hd += hd
                    val_mae += mae
                    val_mbiou += mbiou
                    val_num += 1

            val_acc = val_acc / val_num
            val_pre = val_pre / val_num
            val_rec = val_rec / val_num
            val_miou = val_miou / val_num
            val_dice = val_dice / val_num
            val_hd = val_hd / val_num
            val_mae = val_mae / val_num
            val_mbiou = val_mbiou / val_num

            val_logs['epoch'].append(epoch + 1)
            val_logs['acc'].append(val_acc)
            val_logs['pre'].append(val_pre)
            val_logs['rec'].append(val_rec)
            val_logs['Miou'].append(val_miou)
            val_logs['dice'].append(val_dice)
            val_logs['HD'].append(val_hd)
            val_logs['m_b_iou'].append(val_mbiou)
            val_logs['MAE'].append(val_mae)


            now_metric = val_miou * 0.5 + val_dice * 0.5
            if now_metric > best_val_metric:
                best_val_metric = now_metric
                best_epoch = epoch
                model_data_in_memory = BytesIO()
                torch.save(model.state_dict(),
                           model_data_in_memory, pickle_protocol=-1)
                model_data_in_memory.seek(0)
            # with open(args.save_path, "wb") as f:
            #     f.write(model_data_in_memory.read())
            # model_data_in_memory.close()
            # print(f"Epoch [{fold + 1}/{10}] - Avg. IoU: {mm_b_iou:.4f}")
            # print(f"Epoch [{fold+ 1}/{10}] - Avg. dice: {m_dice:.4f}")
        print("the best epoch:---", best_epoch)

        model_in_cpu = torch.load(model_data_in_memory, map_location="cpu")
        model_data_in_memory.close()
        model.load_state_dict(model_in_cpu)
        model.to("cuda")
        train_df = pd.DataFrame(train_logs)
        train_df.to_csv(os.path.join(args.log_dir, f'{fold+1}_train_log.csv'), index=False)
        val_df = pd.DataFrame(val_logs)
        val_df.to_csv(os.path.join(args.log_dir, f'{fold+1}_val_log.csv'), index=False)
        best_model_path = os.path.join(args.log_dir, f'{fold+1}_{best_epoch}_best_model.pth')
        save_checkpoint(best_model_path, model, key="model")

        #测试部分  加载每一折最好的模型参数 然后进行测试评估 并保存结果
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        model.encoder.load_state_dict(torch.load(best_model_path), strict=False)

        test_acc = 0
        test_pre = 0
        test_rec = 0
        test_miou = 0
        test_dice = 0
        test_hd = 0
        test_mbiou = 0
        test_mae = 0
        test_num = 0
        
        for images, masks in test_loader:
            images = images.to("cuda")
            masks = masks.to("cuda")

            outputs = model(images)
            predicted = outputs

            predicted = predicted.cpu()
            masks = masks.cpu()
            predicted = torch.sigmoid(predicted)

            # predicted = (predicted > 0.5).astype(np.float32)
            # masks = (masks > 0.5).astype(np.float32)
            predicted = np.where(predicted > 0.5, 1, 0)
            masks = np.where(masks > 0.5, 1, 0)
            

            acc, pre, rec, miou, dice, hd, mbiou, mae = evalResult(masks, predicted)
            test_acc += acc
            test_pre += pre
            test_rec += rec
            test_miou += miou
            test_dice += dice
            test_hd += hd
            test_mae += mae
            test_mbiou += mbiou
            test_num += 1

        test_acc = test_acc / test_num
        test_pre = test_pre / test_num
        test_rec = test_rec / test_num
        test_miou = test_miou / test_num
        test_dice = test_dice / test_num
        test_hd = test_hd / test_num
        test_mae = test_mae / test_num
        test_mbiou = test_mbiou / test_num

        test_logs['fold'].append(fold + 1)
        test_logs['acc'].append(test_acc)
        test_logs['pre'].append(test_pre)
        test_logs['rec'].append(test_rec)
        test_logs['Miou'].append(test_miou)
        test_logs['dice'].append(test_dice)
        test_logs['HD'].append(test_hd)
        test_logs['m_b_iou'].append(test_mbiou)
        test_logs['MAE'].append(test_mae)
        test_df = pd.DataFrame(test_logs)
        test_df.to_csv(os.path.join(args.log_dir, f'{fold + 1}_test_log.csv'), index=False)


if __name__ == '__main__':
    main()

