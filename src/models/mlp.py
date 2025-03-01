import torch.nn as nn


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, layers : list[int] = [256,128,64], dropout : bool = False, p : float =0.2):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            layers (list): list that specifies how many hidden layers and how many neurons per hidden layer
            p (float): ratio of dropped out weights
        """
        assert len(layers) != 0 , "MLP has no hidden layers!"
        super(MLP, self).__init__()

        fc = []
        layers = [input_size] + layers + [n_classes]
        for i in range(len(layers) - 1):
            fc.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                fc.append(nn.ReLU())
            if dropout:
                fc.append(nn.Dropout(p))
        self.network = nn.Sequential(*fc)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        preds = x
        preds = self.network(x)
        return preds
