import numpy as np
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from models.dataset import SegmentationDataset
from main import load_model
from utils import dice, IoU, accuracy
from preprocessing.resizing import resize_image

from PIL import Image

clip_train_dim = (352,352)
clip_label_dim = (224,224)
train_dim = (256,256)
label_dim = (256,256)


def main(args):
    model = load_model(args)

    x_y = SegmentationDataset(
        Path(args.train), 
        Path(args.label),
        Path(args.json),
        nn_type=args.nn_type,
        )
    
    train_size = int(len(x_y))
    val_size = len(x_y) - train_size

    x_y, _ = random_split(x_y, [train_size, val_size])
    
    dataloader = DataLoader(x_y, batch_size=1, shuffle=False)
    model.eval()

    acc_scores = []
    IoU_scores = []
    dice_scores = []

    with torch.no_grad():
        #1. Iterate over every image
        for i, batch in enumerate(dataloader):
            if args.nn_type == "CLIP":
                (prompt, x, y, _) = batch
                assert len(prompt) == 1 and len(x) == 1 and len(y) == 1, "Must have batch size of 1"
                prompt = prompt[0]
                x = x[0]
                y = y[0]
            else:
                x, y = batch

            # print(f"Initially y has shape {y.shape}")
            if args.nn_type == "CLIP":
                gt_orig_res = torch.argmax(y, dim=0)
            else:
                gt_orig_res = torch.argmax(y, dim=1)

            x_orig_res_shape = x.shape
            y_orig_res_shape = y.shape

            assert x_orig_res_shape == y_orig_res_shape, "Label and mask must be of same dimensions"

            x = resize_image(
                x, 
                clip_train_dim if args.nn_type == "CLIP" else train_dim,
                InterpolationMode.BILINEAR
            )

            ### Unpack x and y to add a batch dimension

            if args.nn_type == "CLIP":
                torch_to_PIL = transforms.ToPILImage()
                img = torch_to_PIL(x).convert("RGB")
                y_pred = model.forward(img, prompt)
            else:
                y_pred = model.forward(x)

            # print(f"Initially y_pred has shape {y_pred.shape}")
            y_pred_classes = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)  # (N, W, H)
            # y_pred_one_hot = F.one_hot(y_pred_classes, num_classes=3).permute(0, 3, 1, 2)  # (N, 3, W, H)
            
            # print(f"y_pred_classes has shape {y_pred_classes.shape}")
            # y_pred_one_hot = y_pred_one_hot[0]

            y_pred_classes = resize_image(
                y_pred_classes, 
                (y_orig_res_shape[1], y_orig_res_shape[2]) if args.nn_type == "CLIP" else (y_orig_res_shape[2], y_orig_res_shape[3]), 
                InterpolationMode.NEAREST
            )

            # print(f"y_pred_classes has shape {y_pred_classes.shape}")

            # y_pred_classes = y_pred_classes.unsqueeze(0)
            if args.nn_type == "CLIP":
                y = y.unsqueeze(0)
                gt_orig_res = gt_orig_res.unsqueeze(0)

            y_pred_one_hot = F.one_hot(y_pred_classes, num_classes=3).permute(0, 3, 1, 2)  # (N, 3, W, H)

            IoU_score = IoU(y_pred=y_pred_one_hot.cpu().detach().numpy(), y=y.cpu().detach().numpy())
            dice_score = dice(y_pred_one_hot.cpu().detach().numpy(), y.cpu().detach().numpy())
            acc = accuracy(y=gt_orig_res.cpu().detach().numpy(),y_pred=y_pred_classes.cpu().detach().numpy())

            acc_scores.append(acc)
            IoU_scores.append(IoU_score)
            dice_scores.append(dice_score)
            print('\rCalculated performance for sample {}/{}: acc: {:.3f}, IoU : {:.3f}, Dice score: {:.3f}'.
                format(i + 1, len(dataloader), acc, IoU_score, dice_score), end='')
            

    print(f"Validation accuracy = {np.array(acc_scores).mean()}")
    print(f"Validation IoU = {np.array(IoU_scores).mean()}")
    print(f"Validation Dice = {np.array(dice_scores).mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=None, type=str, help="path to directory with train images of original **unresized** dimensions")
    parser.add_argument('--label', default=None, type=str, help="path to directory with label images of original **unresized** dimensions")
    parser.add_argument('--json', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--load',default=None, type=str, help="path to .pth file with model parameters")
    parser.add_argument('--nn_type',default="CLIP", type=str, help="type of Neural Net being loaded")
    parser.add_argument('--nn_batch_size', default = 1, type=int)
    
    args = parser.parse_args()
    if not(args.train) or not(args.label):
        raise ValueError("Must input train and label path if resize/augment of images wanted or json map")
    
    main(args)