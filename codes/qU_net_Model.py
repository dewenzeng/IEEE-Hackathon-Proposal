from collections import OrderedDict
import torch
import torch.nn as nn
from layers import *

kernel_size = 3


class qUNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, config=''):
        super(qUNet, self).__init__()

        self.config=config
        features = init_features
        self.encoder1 = qUNet._qblock(in_channels, features, name="enc1", config=self.config)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = qUNet._qblock(features, features * 2, name="enc2", config=self.config)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = qUNet._qblock(features * 2, features * 4, name="enc3", config=self.config)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = qUNet._qblock(features * 4, features * 8, name="enc4", config=self.config)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = qUNet._qblock(features * 8, features * 16, name="bottleneck", config=self.config)

        self.upconv4 = ConvTranspose2dQuant(
            features * 16, features * 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), config=self.config
        )
        self.decoder4 = qUNet._qblock((features * 8) * 2, features * 8, name="dec4", config=self.config)
        self.upconv3 = ConvTranspose2dQuant(
            features * 8, features * 4, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), config=self.config
        )
        self.decoder3 = qUNet._qblock((features * 4) * 2, features * 4, name="dec3", config=self.config)
        self.upconv2 = ConvTranspose2dQuant(
            features * 4, features * 2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), config=self.config
        )
        self.decoder2 = qUNet._qblock((features * 2) * 2, features * 2, name="dec2", config=self.config)
        self.upconv1 = ConvTranspose2dQuant(
            features * 2, features, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), config=self.config
        )
        self.decoder1 = qUNet._qblock(features * 2, features, name="dec1", config=self.config)

        self.conv = Conv2dQuant(
            in_channels=features, out_channels=out_channels, kernel_size=1, config=self.config
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1) #torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _qblock(in_channels, features, name, config=''):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        Conv2dQuant(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            config=config,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "quant1", QuantLayer(config=config)),
                    (
                        name + "conv2",
                        Conv2dQuant(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            config=config,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "quant2", QuantLayer(config=config)),
                ]
            )
        )


