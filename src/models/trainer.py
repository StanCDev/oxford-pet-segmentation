import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np

from utils import IoU, accuracy

red_scale = 1/3 * 1/0.103
green_scale = 1/3* 1/0.194
background_scale = 1/3*1/0.704


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(
            self, 
            model, 
            lr, 
            epochs, 
            nn_type : str ,
            batch_size,device = torch.device("cpu"), 
            weight : torch.Tensor = torch.tensor([red_scale, green_scale, background_scale], dtype=torch.float32),
            ):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.nn_type = nn_type

        weight = weight.to(device)

        self.criterion = None
        if nn_type == "autoencoder":
            self.criterion = nn.MSELoss()
        else:
            self.criterion : nn.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
        
        self.optimizer = torch.optim.Adam(params=model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8)
        self.device = device

        self.loss_mu = []
        self.loss_sigma = []
        self.IoU_mu = []
        self.IoU_sigma = []
        self.acc_mu = []
        self.acc_sigma = []
        # ...

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        epochs = range(self.epochs)
        for ep in epochs:
            self.train_one_epoch(dataloader,ep, len(epochs))
            print("")

    def train_one_epoch(self, dataloader, ep, epochs):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        # Statistics
        temp_loss = []
        temp_IoU = []
        temp_acc = []
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            if self.nn_type == "CLIP":
                (prompt, x, y, _) = batch
                assert len(prompt) == 1 and len(x) == 1 and len(y) == 1, "Must have batch size of 1"
                # print(f"Here are the types of prompt, image, label : {type(prompt)}, {type(image)}, {type(label)}")
                prompt = prompt[0]
                # image = image[0]
                # label = label[0]
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

            loss = None
            x_pred = None
            ground_truths = None
            logits = None

            if self.nn_type == "autoencoder":
                x_pred = self.model.forward(x)
                loss = self.criterion(x_pred, y)
            elif self.nn_type == "CLIP":
                torch_to_PIL = transforms.ToPILImage()
                # 5.2 Run forward pass.
                logits = self.model.forward(torch_to_PIL(x[0]).convert("RGB"), prompt)
                ground_truths = torch.argmax(y, dim=1)
                # 5.3 Compute loss (using 'criterion').
                loss = self.criterion(logits,ground_truths)
            else:
                # 5.2 Run forward pass.
                logits = self.model.forward(x)
                ground_truths = torch.argmax(y, dim=1)
                # 5.3 Compute loss (using 'criterion').
                loss = self.criterion(logits,ground_truths)
            
            # 5.4 Run backward pass.
            loss.backward()
            
            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()
            
            # 5.6 Zero-out the accumulated gradients.
            self.model.zero_grad()

            #5.7 Save loss and iteration number
            temp_loss.append(loss.data.cpu().numpy())

            #5.8
            acc = 0
            IoU_score = 0

            if self.nn_type == "autoencoder":
                y_eval = y.cpu().detach().numpy()
                y_pred_eval = x_pred.cpu().detach().numpy()

                IoU_score = IoU(y=y_eval, y_pred=y_pred_eval)
                acc = accuracy(y=y_eval,y_pred=y_pred_eval)
            else:
                y_pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (N, W, H)
                y_pred_one_hot = F.one_hot(y_pred_classes, num_classes=3).permute(0, 3, 1, 2)  # (N, 3, W, H)

                IoU_score = IoU(y_pred_one_hot.cpu().detach().numpy(), y.cpu().detach().numpy())
                acc = accuracy(y=ground_truths.cpu().detach().numpy(),y_pred=y_pred_classes.cpu().detach().numpy())
            
            temp_IoU.append(IoU_score)
            temp_acc.append(acc)
            print('\rEp {}/{}, it {}/{}: loss train: {:.3f}, IoU train: {:.3f}, accuracy train: {:.3f}'.
                format(ep + 1, epochs, it + 1, len(dataloader), loss,
                        IoU_score, acc), end='')
        ### mean and stdev
        temp_IoU = np.array(temp_IoU)
        temp_loss = np.array(temp_loss)
        temp_acc = np.array(temp_acc)
        self.loss_mu.append(temp_loss.mean())
        self.loss_sigma.append(temp_loss.std())
        self.IoU_mu.append(temp_IoU.mean())
        self.IoU_sigma.append(temp_IoU.std())
        self.acc_mu.append(temp_acc.mean())
        self.acc_sigma.append(temp_acc.std())

    def predict_torch(self, dataloader, display_metrics : bool = False):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []
        acc = []
        iou = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                if self.nn_type == "CLIP":
                    (prompt, x, y, _) = batch
                    assert len(prompt) == 1 and len(x) == 1 and len(y) == 1, "Must have batch size of 1"
                    prompt = prompt[0]
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)

                if self.nn_type == "CLIP":
                    torch_to_PIL = transforms.ToPILImage()
                    # 5.2 Run forward pass.
                    logits = self.model.forward(torch_to_PIL(x[0]).convert("RGB"), prompt)
                    ground_truths = torch.argmax(y, dim=1)
                else:
                    # 5.2 Run forward pass.
                    logits = self.model.forward(x)
                    ground_truths = torch.argmax(y, dim=1)

                y_pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (N, W, H)
                y_pred_one_hot = F.one_hot(y_pred_classes, num_classes=3).permute(0, 3, 1, 2)  # (N, 3, W, H)

                pred_labels.append(y_pred_classes) ### want to take the max along channels

                if display_metrics:
                    IoU_score = IoU(y_pred_one_hot.cpu().detach().numpy(), y.cpu().detach().numpy())
                    acc_score = accuracy(y=ground_truths.cpu().detach().numpy(),y_pred=y_pred_classes.cpu().detach().numpy())
                    acc.append(acc_score)
                    iou.append(IoU_score)
        
        if display_metrics:
            print(f"Validation accuracy = {np.array(acc).mean()}")
            print(f"Validation IoU = {np.array(iou).mean()}")
        return torch.cat(pred_labels)
    
    def fit(self, training_data : Dataset):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (Dataset): training data and labels
        Returns:
            pred_labels (array): target of shape (N,)
        """
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data : Dataset, display_metrics : bool = False):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (D): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader, display_metrics=display_metrics)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()