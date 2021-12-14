import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TODO: build the model here
class PilotNet(nn.Module):
    """Implementation of PilotNet. See the model here: https://miro.medium.com/max/500/1*VB_OYZu4DDlNIT7mdSstcg.png"""

    def __init__(self, in_channels=3, out_channels=1):
        """

        Args:
            in_channels (int, optional): number of channels for the input (3 for RGB frames, 1 for grayscale frames, ...). Defaults to 3.
            out_channels (int, optional): Number of channels for the output (1 for the angle prediction, 2 for angle+throttle prediction, ...). Defaults to 1.
        """
        super(PilotNet, self).__init__()

        # TODO: declare your model here
        pass

    def forward(self, frame):

        return None  # TODO: output must be the angle


# Utility layers (don't hesitate to use it)
class ConvBNAct(nn.Sequential):
    """Convolution + BatchNorm + activation combo"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, activation=torch.nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv', nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,  # no bias when using batchNorm because batchnorm adds the bias
                                          dilation=dilation,
                                          stride=stride))

        self.add_module('bn', nn.BatchNorm2d(out_channels))

        self.add_module('activation', activation)


class ConvBN(nn.Sequential):
    """Conv + Batch norm combo"""

    def __init__(self, channels_in, channels_out, kernel_size):
        super(ConvBN, self).__init__()
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2,
                                          bias=False))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
