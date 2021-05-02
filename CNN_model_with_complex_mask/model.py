import torch
import torch.nn as nn
import torch.nn.functional as functional
import os


class AudioDenoiserNet(nn.Module):
    def __init__(self, sequence_length):
        super(AudioDenoiserNet, self).__init__()
        assert sequence_length > 0
        self.sequence_length = sequence_length
        self.numSegments = 129
        self.linear_unit = self.numSegments * 2 # "2" corresponds to the real & imaginary channel
        self.name = "AudioDenoiser"
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=18, kernel_size=(9, sequence_length),  stride=(1, 1), padding=(4,0)),
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
        self.layer16 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=(1, 1),  stride=(1, 1), padding=0)
        self.fc1 = nn.Linear(self.linear_unit,self.linear_unit)
        self.complex_mask_layer = nn.Linear(self.linear_unit,self.linear_unit)



    def forward(self, input_spectrogram):
        x = self.layer1(input_spectrogram)
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
        x = x.view(-1,self.linear_unit)
        x = functional.relu(self.fc1(x))
        x = functional.tanh(self.complex_mask_layer(x))
        x = x.view(-1,input_spectrogram.shape[1],input_spectrogram.shape[2],1)
        output_spectrogram = torch.mul(input_spectrogram[:,:,:,-1:], x)
        return output_spectrogram
