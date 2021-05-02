import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt

from dataset import AudioDataset
from utils import *
from model import AudioDenoiserNet


def get_model_name(model_name, batch_size, learning_rate, epoch):
    return str(model_name) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epoch) + ".pt"


def train_model(model, train_data_loader, valid_data_loader, csv_base, batch_size=32, learning_rate=0.001, num_epochs=5):
    torch.manual_seed(1000)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = np.zeros(num_epochs)
    valid_loss = np.zeros(num_epochs)

    start_time = time.time()
    batch_number = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        batch_number = 0

        for data in train_data_loader:
            inputs, labels = data
            count += len(data)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            # Save loss per batch
            csv_base.save_loss_per_batch(epoch, batch_number, loss.item())

            # input_snr = 0
            # output_snr = 0

            # # Save input output snr difference
            # for batch_index in range(inputs.shape[0]):
            #     input_snr += calculate_input_snr(labels[batch_index], inputs[batch_index])
            #     output_snr += calculate_output_snr(labels[batch_index], pred[batch_index])

            # input_snr = input_snr/batch_size
            # output_snr = output_snr/batch_size
            # csv_base.save_input_output_snr_difference(epoch, batch_number, input_snr, output_snr)

            # sum_input_noise = 0
            # sum_output_noise = 0

            # # Save log ratio of output input noise
            # for batch_index in range(inputs.shape[0]):
            #     sum_input_noise += calculate_input_noise_sum(labels[batch_index], inputs[batch_index])
            #     sum_output_noise += calculate_output_noise_sum(labels[batch_index], pred[batch_index])

            # sum_input_noise = sum_input_noise/batch_size
            # sum_output_noise = sum_output_noise/batch_size
            # csv_base.save_log_ratio_output_input_noise(epoch, batch_number, math.log10(sum_output_noise/sum_input_noise))

            batch_number += 1
            running_loss += loss.item()
            print("Current Loss: ", loss.item())
            print("Running Loss: ", running_loss)

        # Save loss per epoch
        csv_base.save_loss_per_epoch(epoch, running_loss)

        train_loss[epoch] = float(running_loss)/(count)
        valid_loss[epoch] = evaluate(model,valid_data_loader,criterion)
        # Save the current model (checkpoint) to a file every 10 epochs
        if epoch % 2 == 0:
            model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)

    end_time = time.time()
    print("training time = ",end_time-start_time,"s")
    model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
    torch.save(model.state_dict(), model_path)    
    return train_loss, valid_loss


def evaluate(model,data_loader,criterion):
    model.eval()
    total_loss = 0
    for data in data_loader:
        inputs, target = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, target)

        total_loss += loss.item()

    return float(total_loss)/(len(data_loader))

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


def main(csv_base):
    sequence_length = 8
    model = AudioDenoiserNet(sequence_length)
    # weight initialization
    model = model.apply(weight_init)

    if torch.cuda.is_available():
        model.cuda()

    batch_size = 512
    learning_rate = 0.001
    num_epochs = 30

    data_dir = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_dataset2/old_data"
    saving_figure = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/results2/"

    train_dataset = AudioDataset(data_dir,"training")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    val_dataset = AudioDataset(data_dir,"validation")
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    train_loss, valid_loss = train_model(model, train_data_loader, val_data_loader, csv_base, batch_size, learning_rate, num_epochs)
    plot_loss(train_loss,valid_loss,saving_figure)

if __name__ == "__main__":

    loss_per_epoch_csv = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/results2/loss_per_epoch.csv"
    loss_per_batch_csv = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/results2/loss_per_batch.csv"
    input_output_snr_difference_csv = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/results2/input_output_snr_difference.csv"
    log_ratio_output_input_noise_csv = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/results2/log_ratio_output_input_noise.csv"

    csv_base = csv_generator(loss_per_epoch_csv,loss_per_batch_csv,input_output_snr_difference_csv,log_ratio_output_input_noise_csv)

    main(csv_base)


# # TODO:
# Instead of using magnitude, use the spectrogram with two channels (real and imaginary) to train
# utilizes the same idea in speech separation model, use a complex mask instead of directly generating the predicted spectrogram