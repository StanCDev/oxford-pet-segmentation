import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from ..utils import IoU


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size,device = torch.device("cpu")):
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

        self.criterion = nn.CrossEntropyLoss() ##CHANGE THIS
        self.optimizer = torch.optim.Adam(model.parameters(), lr,weight_decay=5*1e-3)
        ###torch.optim.RAdam()
        self.device = device

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
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y = y.long()
            # 5.2 Run forward pass.
            predicted = self.model.forward(x)
            
            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(predicted,y)
            
            # 5.4 Run backward pass.
            loss.backward()
            
            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()
            
            # 5.6 Zero-out the accumulated gradients.
            self.model.zero_grad()

            print('\rEp {}/{}, it {}/{}: loss train: {:.2f}, accuracy train: {:.2f}'.
                format(ep + 1, epochs, it + 1, len(dataloader), loss,
                        IoU(predicted, y)), end='')

    def predict_torch(self, dataloader):
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
        with torch.no_grad():
            for it, x in enumerate(dataloader):
                x = x[0]
                x = x.to(self.device)
                y = self.model(x)
                pred_labels.append(torch.argmax(y, dim=1)) ### change here
        return torch.cat(pred_labels)
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()