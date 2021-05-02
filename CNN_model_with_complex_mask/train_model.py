import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import AudioDataset
from utils import *
from model import AudioDenoiserNet
import config as cfg


def get_model_name(model_name, batch_size, learning_rate, epoch):
    return str(model_name) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epoch) + ".pt"


def train_model(model, train_data_loader, valid_data_loader, log_dir, batch_size=32, learning_rate=0.001, num_epochs=5):
    torch.manual_seed(1000)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)
    train_loss = np.zeros(num_epochs)
    valid_loss = np.zeros(num_epochs)

    start_time = time.time()
    step = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0

        for data in tqdm(train_data_loader):
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
            running_loss += loss.item()

        epoch_training_loss = float(running_loss)/(count)
        epoch_validation_loss = evaluate(model,valid_data_loader,criterion)
        train_loss[epoch] = epoch_training_loss
        valid_loss[epoch] = epoch_validation_loss

        writer.add_scalar('Loss/Train', epoch_training_loss, global_step=step)
        writer.add_scalar('Loss/Validation', epoch_validation_loss, global_step=step)
        step += 1

        # Save the current model (checkpoint) to a file every 10 epochs
        if epoch % 2 == 0:
            model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)

    end_time = time.time()
    print("training time = ",end_time-start_time,"s")
    model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
    torch.save(model.state_dict(), model_path)

    torch.save(model,'full_model_'+model_path)
    return train_loss, valid_loss

def hyperparameter_search(model, train_data_loader, valid_data_loader, log_dir, num_epochs=5):
    torch.manual_seed(1000)
    criterion = nn.MSELoss()

    batch_sizes = [128,256,512,1024]
    learning_rates = [0.001,0.0001,0.0005,0.00001]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_loss = np.zeros(num_epochs)
            valid_loss = np.zeros(num_epochs)
            writer = SummaryWriter(log_dir=log_dir+'Batchsize{batch_size}LR{lr}')
            step = 1
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                count = 0

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
                    running_loss += loss.item()

                epoch_training_loss = float(running_loss)/(count)
                epoch_validation_loss = evaluate(model,valid_data_loader,criterion)
                train_loss[epoch] = epoch_training_loss
                valid_loss[epoch] = epoch_validation_loss

                writer.add_scalar('Loss/Train', epoch_training_loss, global_step=step)
                writer.add_scalar('Loss/Validation', epoch_validation_loss, global_step=step)

                writer.add_hparams({'lr': lr, 'bsize': batch_size},{'train_loss': epoch_training_loss,'valid_loss': epoch_validation_loss})
                step += 1

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
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)



def main(log_dir, results_path):

    model = AudioDenoiserNet(cfg.SEQUENCE_LENGTH)
    # weight initialization
    model = model.apply(weight_init)

    if torch.cuda.is_available():
        model.cuda()

    batch_size = cfg.BATCH_SIZE
    learning_rate = cfg.LEARNING_RATE
    num_epochs = cfg.NUM_EPOCHS

    train_dataset = AudioDataset(cfg.DATA_DIR,"training")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    val_dataset = AudioDataset(cfg.DATA_DIR,"validation")
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # hyperparameter_search(model, train_data_loader, val_data_loader, log_dir, num_epochs)
    train_loss, valid_loss = train_model(model, train_data_loader, val_data_loader, log_dir, batch_size, learning_rate, num_epochs)
    # plot_loss(train_loss,valid_loss,results_path)


if __name__ == "__main__":

    # Paths
    data_dir = cfg.DATA_DIR
    results = cfg.RESULTS
    log_dir = cfg.LOG_DIR

    if not os.path.isdir(results):
        os.makedirs(results)

    # loss_per_epoch_csv = os.path.join(results,"loss_per_epoch.csv")
    # loss_per_batch_csv = os.path.join(results,"loss_per_batch.csv")
    # input_output_snr_difference_csv = os.path.join(results,"input_output_snr_difference.csv")
    # log_ratio_output_input_noise_csv = os.path.join(results,"log_ratio_output_input_noise.csv")

    # Save training statistics
    # csv_base = csv_generator(loss_per_epoch_csv,loss_per_batch_csv,input_output_snr_difference_csv,log_ratio_output_input_noise_csv)

    main(log_dir,results)