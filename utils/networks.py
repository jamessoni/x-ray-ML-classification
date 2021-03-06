"""
This module contains network architectures e.g. LeNet5, AlexNet, ResNet etc.
The user can place architectures here so that they can be easily imported
into the notbook.
This modules purpose is to have all the modules we have used self contained
in one location.

Architectures placed in here should follow this structure:

class ExampleNet(nn.Module):
    def __init__(self):
        super(Examplenet, self).__init__()
        ***********************
        Network layers go here
        ***********************

    def forward(self, x):
        *****************
        Define the forward pass
        *****************
        return self.output(x) <--- output should be the last layer.

-------------------------------------------------------------------
To import a network in the jupyter notebook use:

                            from networks import <ExampleNet>

"""

import torch.nn as nn


class LeNet5(nn.Module):
    """
    Custom class defining the LeNet-5 architecture.
    Implements two functions, __init__ constructor function
    defining convolution and down-sampling operations and
    the forward function.

    Attributes
    ----------
    c: torch.nn.modules.conv.Conv2d
        Applies a 2D convolution over an input signal composed
            of several input planes
    s: torch.nn.modules.pooling.MaxPool2d
        Applies a 2D max pooling over an input signal composed
            of several input planes
    f: torch.nn.modules.linear.Linear
        Applies a linear transformation to the incoming data
    output: torch.nn.modules.linear.Linear
        Applies a linear transformation with the required output channels
    act: torch.nn.modules.activation.ReLU
        Applies a rectified linear unit function

    Methods
    -------
    forward(x)
        Forward pass using the layers and activations of the class

    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Linear(16 * 72 * 72, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 4)  # change to 4
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass using the layers and activations of the class.

        Parameters
        ----------
        x: tensor
            flattened or reshaped tensor matching the input dimensions
                required for the fully connect block of neural network

        Returns
        -------
        tensor
            Forwarded tensor

        """
        x = self.act(self.c1(x))
        x = self.act(self.s2(x))
        x = self.act(self.c3(x))
        x = self.act(self.s4(x))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.act(self.c5(x))
        x = self.act(self.f6(x))
        return self.output(x)


class AlexNet(nn.Module):
    """
    Our first more "succesful" network, slightly modified AlexNet
    that accepts images in with 1 channel (i.e. grayscale).

    Attributes
    ----------
    c: torch.nn.modules.conv.Conv2d
        Applies a 2D convolution over an input signal composed
            of several input planes
    s: torch.nn.modules.pooling.MaxPool2d
        Applies a 2D max pooling over an input signal composed
            of several input planes
    avgpool: torch.nn.modules.pooling.AdaptiveAvgPool2d
        Applies a 2D adaptive average pooling over an input signal composed
            of several input planes
    d: torch.nn.modules.dropout.Dropout
        Randomly zeroes some elements of the input tensor during training
            using a Bernoulli distribution
    f: torch.nn.modules.linear.Linear
        Applies a linear transformation to the incoming data
    output: torch.nn.modules.linear.Linear
        Applies a linear transformation with the required output channels
    act: torch.nn.modules.activation.ReLU
        Applies a rectified linear unit function

    Methods
    -------
    forward(x)
        Forward pass using the layers and activations of the class

    """
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.d9 = nn.Dropout()
        self.f10 = nn.Linear(256 * 6 * 6, 4096)
        self.d11 = nn.Dropout()
        self.f12 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass using the layers and activations of the class.

        Parameters
        ----------
        x: tensor
            flattened or reshaped tensor matching the input dimensions
                required for the fully connect block of neural network

        Returns
        -------
        tensor
            Forwarded tensor

        """
        x = self.act(self.c1(x))
        x = self.act(self.s2(x))
        x = self.act(self.c3(x))
        x = self.act(self.s4(x))
        x = self.act(self.c5(x))
        x = self.act(self.c6(x))
        x = self.act(self.c7(x))
        x = self.act(self.s8(x))

        x = self.avgpool(x)

        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.act(self.d9(x))
        x = self.act(self.f10(x))
        x = self.act(self.d11(x))
        x = self.act(self.f12(x))
        return self.output(x)
