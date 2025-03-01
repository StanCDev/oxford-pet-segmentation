import argparse

import numpy as np
import torch
from torchinfo import summary
from models.trainer import Trainer
from models.unet import UNet

seed = 100

np.random.seed(seed) ### Randomness for cross val


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                        of this file). Their value can be accessed as "args.argument".
    """
    ## 1. Load data
    xtrain = None
    ytrain = None
    xval = None
    yval = None
    xtest = None
    ytest = None

    ## 2. Make a validation set
    if not args.test:
        N = xtrain.shape[0]
        all_ind = np.arange(N)
        split_ratio = 0.2
        split_size = int(split_ratio * N)

        ################### RANDOM SHUFFLING ################
        all_ind = np.random.permutation(all_ind)
        #####################################################

        ########### TRAINING AND VALIDATION INDICES #########
        val_ind = all_ind[: split_size]
        train_ind = np.setdiff1d(all_ind, val_ind, assume_unique=True)
        #####################################################

        xtrain_original = xtrain
        ytrain_original = ytrain

        xtrain = xtrain_original[train_ind]
        xtest = xtrain_original[val_ind]

        xval = ytrain_original[val_ind]
        yval = ytrain_original[train_ind]


    ## 3. Selecting device
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
        model = UNet(0,0,0)
    else:
        raise ValueError("Inputted model is not a valid model")

    summary(model)

    model.to(device)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size,device=device)


    ## 4. Train and evaluate the method
    preds_train = method_obj.fit(xtrain, ytrain)
    # Predict on unseen data
    preds = method_obj.predict(xval)

    ## 5. Evaluation metrics

    ## 6. Saving the model
    if args.save != "NONE":
        torch.save(model.state_dict(), args.save)
    np.save("predictions", preds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="unet",
                        help="which network architecture to use, it can be 'unet' | 'autoencoder' | 'CLIP' | 'prompt'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")

    ### Pytorch saving / loading models
    parser.add_argument('--save', default="NONE", type=str, help="path where you want to save your model")
    parser.add_argument('--load', default="NONE", type=str, help="path where you want to load your model")

    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")

    args = parser.parse_args()
    main(args)