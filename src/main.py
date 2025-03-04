import argparse

import torch
from torchinfo import summary

from pathlib import Path

from models.trainer import Trainer
from models.unet import UNet
from models.dataset import SegmentationDataset
from torch.utils.data import random_split

from utils import seed_everything

seed = 100


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                        of this file). Their value can be accessed as "args.argument".
    """
    ## 0. seed randomness
    seed_everything(seed=seed)

    ## 1. Load data
    x_y_train = SegmentationDataset(
        Path("/Users/stancastellana/Desktop/UoE/Ba6/Computer_Vision/MP/Dataset/Processed/train"), 
        Path("/Users/stancastellana/Desktop/UoE/Ba6/Computer_Vision/MP/Dataset/Processed/label"),
        Path("/Users/stancastellana/Desktop/UoE/Ba6/Computer_Vision/MP/CV_mini_project/res/mapping.json")
        )
    x_y_val = None
    x_y_test = None

    ## 2. Make a validation set
    train_size = int(0.8 * len(x_y_train))
    val_size = len(x_y_train) - train_size

    x_y_train, x_y_val = random_split(x_y_train, [train_size, val_size])


    ## 3. Selecting device (CPU/GPU)
    device = torch.device("cpu")
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("cuda not available on this device")
    elif args.device == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            print("mps not available on this device")


    ## 3. Initialize the method you want to use.
    model = None
    if args.nn_type == "unet":
        model = UNet(w=256,h=256,ch=3, ch_mult=8)
    else:
        raise ValueError("Inputted model is not a valid model")
    
    if args.load is not None:
        model.load_state_dict(torch.load(args.load, weights_only=True))

    summary(model)

    model.to(device)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size,device=device)


    ## 4. Train and evaluate the method
    preds_train = method_obj.fit(x_y_train)
    # Predict on unseen data
    # preds = method_obj.predict(xval)

    ## 5. Evaluation metrics

    ## 6. Saving and loading the model
    if args.save is not None:
        torch.save(model.state_dict(), args.save)
    # np.save("predictions", preds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nn_type', default="unet",
                        help="which network architecture to use, it can be 'unet' | 'autoencoder' | 'CLIP' | 'prompt'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")

    ### Pytorch saving / loading models
    parser.add_argument('--save', default=None, type=str, help="path where you want to save your model")
    parser.add_argument('--load', default=None, type=str, help="path where you want to load your model")

    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")

    args = parser.parse_args()
    main(args)