import argparse

import torch
from torchinfo import summary

from pathlib import Path

from models.trainer import Trainer
from models.unet import UNet
from models.autoencoder import AutoEncoder
from models.clip_seg import Clip
from models.dataset import SegmentationDataset
from torch.utils.data import random_split

from utils import seed_everything, plot_training_metrics
seed = 100

def load_model(args):
    model = None
    if args.nn_type == "unet":
        model = UNet(w=256,h=256,ch=3, ch_mult=8)
    
    elif args.nn_type == "autoencoder":
        model = AutoEncoder(w=256,h=256,in_channels=3,out_channels=3, ch_mult=4)
    
    elif args.nn_type == "autoencoder_segmentation":
        if args.load is None:
            raise ValueError("If nn_type is autoencoder_segmentation then --load must be specified as a loaded autoencoder needs to exist")
        pre_trained = AutoEncoder(w=256,h=256,in_channels=3,out_channels=3, ch_mult=4)
        pre_trained.load_state_dict(torch.load(args.load, weights_only=True))
        
        model = AutoEncoder(w=256,h=256,in_channels=3,out_channels=3, ch_mult=4)
        model.encoder = pre_trained.encoder

        for param in model.encoder.parameters():
            param.requires_grad = False  # No gradients for encoder
    
    elif args.nn_type == "CLIP":
        if args.nn_batch_size != 1:
            raise ValueError("Batch size for CLIP must be 1")
        model = Clip()
    else:
        raise ValueError("Inputted model is not a valid model")
    
    if args.load is not None:
        model.load_state_dict(torch.load(args.load, weights_only=True))
    return model


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
    base_path : Path = Path(args.base_path)
    base_path_test : Path = Path(args.base_path_test)

    json_path = Path(args.json)
    json_path_test = Path(args.json_test)

    x_y_train = SegmentationDataset(
        Path(base_path / "train"), 
        Path(base_path / "label"),
        json_path,
        nn_type=args.nn_type,
        )
    x_y_test = None
    if args.test:
        x_y_test = SegmentationDataset(
            Path(base_path_test / "train"),
            Path(base_path_test / "label"),
            json_path_test,
            nn_type=args.nn_type,
        )

    ## 2. Make a validation set
    if not(args.cv_ratio >= 0 and args.cv_ratio <= 1):
        raise ValueError(f"cv_ratio must be between 0 and 1. Input = {args.cv_ratio}")

    train_size = int(args.cv_ratio * len(x_y_train))
    val_size = len(x_y_train) - train_size

    generator = torch.Generator().manual_seed(seed)
    x_y_train, x_y_val = random_split(x_y_train, [train_size, val_size],generator)


    ## 3. Selecting device (CPU/GPU)
    ## add restriction wrt to clip?????
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
    model = load_model(args)
    summary(model)

    model.to(device)

    # Trainer object
    method_obj = Trainer(model=model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size,device=device,nn_type=args.nn_type)


    ## 4. Train and evaluate the method
    preds_train = None
    if args.train:
        if args.evaluate_val:
            preds_train = method_obj.fit(x_y_train, x_y_val)
        else:
            preds_train = method_obj.fit(x_y_train)

    ## 5. predict on unseen data
    if not args.evaluate_val and not args.test:
        preds_val = method_obj.predict(x_y_val, display_metrics = True, nn_save_output=False)

    if args.test:
        print("Test dataset metrics:")
        preds_test = method_obj.predict(x_y_test, display_metrics = True, nn_save_output=True)

    ## 6. Saving and loading the model
    if args.save is not None:
        torch.save(model.state_dict(), args.save)
    if args.train:
        plot_training_metrics(method_obj, show_val=args.evaluate_val)
    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nn_type', default="unet",
                        help="which network architecture to use, it can be 'unet' | 'autoencoder' | 'autoencoder_segmentation' | 'CLIP' Note that CLIP only works with a batch size of 1")
    parser.add_argument('--prompt',action="store_true",default=False, help="Train on prompt dataset")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")

    ### Pytorch saving / loading models
    parser.add_argument('--save', default=None, type=str, help="path where you want to save your model")
    parser.add_argument('--load', default=None, type=str, help="path where you want to load your model")

    parser.add_argument('--train',action="store_true",default=False, help = "Train model")
    parser.add_argument('--test',action="store_true",default=False,help="Evaluate model on test dataset")
    parser.add_argument('--evaluate_val', action="store_true", help = "At every epoch, compute and store metrics on validation set")

    ### Paths to datasets
    parser.add_argument('--json',action="store_true",default=False, help = "Path to json file that contains mapping from into to data point (image) for training dataset, generated in preprocessing steps")
    parser.add_argument('--json_test',action="store_true",default=False, help = "Path to json file that contains mapping from into to data point (image) for test dataset, generated in preprocessing steps")
    parser.add_argument('--base_path',action="store_true",default=False, help = "Base directory for training data that must contain two sub directories train and label.")
    parser.add_argument('--base_path_test',action="store_true",default=False, help = "Base directory for test data that must contain two sub directories train and label.")

    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")

    parser.add_argument('--cv_ratio', type=float, default=0.8,help="ratio of data for train data set. 1 - cv_ratio is ratio of data for validation set. Value must be between 0 and 1")

    args = parser.parse_args()
    main(args)