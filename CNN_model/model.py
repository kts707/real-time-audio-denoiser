import torch
import torch.nn as nn
import torch.nn.functional as functional
import os


class AudioDenoiserNet(nn.Module):
    def __init__(self, sequence_length):
        super(AudioDenoiserNet, self).__init__()
        assert sequence_length > 0
        self.name = "AudioDenoiser"
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=18, kernel_size=(9, sequence_length),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(18),
            nn.ReLU())
        self.skip0 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(5, 1),  stride=(1, 1), padding=(2,0)))
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(30),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=8, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=18, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(18),
            nn.ReLU())
        self.skip1 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(5, 1),  stride=(1, 1), padding=(2,0)))
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(30),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=8, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=18, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(18),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(5, 1),  stride=(1, 1), padding=(2,0)),
            nn.BatchNorm2d(30),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=8, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=18, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(18),
            nn.ReLU())
        self.skip_add1 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(5, 1),  stride=(1, 1), padding=(2,0)))
        self.layer11 = nn.Sequential(
            nn.BatchNorm2d(30),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=8, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=18, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(18),
            nn.ReLU())
        self.skip_add0 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(5, 1),  stride=(1, 1), padding=(2,0)))
        self.layer14 = nn.Sequential(
            nn.BatchNorm2d(30),
            nn.ReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=8, kernel_size=(9, 1),  stride=(1, 1), padding=(4,0)),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.dropout = nn.Dropout2d(p=0.2)
        self.layer16 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1),  stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.layer1(x)
        skip0 = self.skip0(x)
        x = self.skip0(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        skip1 = self.skip1(x)
        x = self.skip1(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.skip_add1(x)
        x = self.layer11(x+skip1)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.skip_add0(x)
        x = self.layer14(x+skip0)
        x = self.layer15(x)
        x = self.dropout(x)
        x = self.layer16(x)
        return x
