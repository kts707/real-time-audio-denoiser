import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import math
import csv
import os
import random
import json
import warnings
import shutil

from process_raw_dataset import FeatureExtractor, read_audio
from model import AudioDenoiserNet

# ------------------------------------------------------------------------------------------------------------------------------------------
# Helper functions for generating training data from noisy spectrogram
def prepare_input_features(noisy_stft_features, clean_stft_features, numSegments, numFeatures):
    '''
    This function prepares the input features for training

    Args:

        numSegments = number of time stamp per input sample
        numFeatures = number of frequency bins for each time stamp
    '''

    # noisySTFT = np.concatenate([noisy_stft_features[:, 0:numSegments - 1], noisy_stft_features], axis=1)

    # stftSegments = np.zeros((numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))
    # stftSegments = np.zeros((noisySTFT.shape[1] - numSegments + 1,numFeatures, numSegments))
    stftSegments = np.zeros((noisy_stft_features.shape[1] - numSegments + 1,numFeatures, numSegments))
    # for index in range(noisySTFT.shape[1] - numSegments + 1):
    #     stftSegments[:, :, index] = noisySTFT[:, index:index + numSegments]

    for index in range(stftSegments.shape[0]):
        stftSegments[index, :, :] = noisy_stft_features[:, index:index + numSegments]

    # print("final shape = ", stftSegments.shape)
    # print("final clean shape = ",clean_stft_features.shape)
    # stftSegments = np.transpose(stftSegments, (2, 0, 1))
    stftSegments = np.expand_dims(stftSegments, axis=1)
    # print("final1 shape = ", stftSegments.shape)

    clean_stft_features = clean_stft_features[:,7:]
    clean_stft_features = np.transpose(clean_stft_features, (1, 0))
    clean_stft_features = np.expand_dims(clean_stft_features, axis=[1,3])

    # print("noisy_stft_features has shape:", stftSegments.shape)
    # print("clean_stft_features has shape:", clean_stft_features.shape)
    return stftSegments, clean_stft_features

# ----------------------------------------------------------------------------------------------------------------------------------------
# Helpers for getting qualitative results for test set given json files of clean audio and noisy audio

def generate_noisy_waveform_for_test_set(clean_json_file, noise_json_file, fs, noisy_saving_path_base, clean_saving_path_base, num_files = None):
    '''
    generate noisy clips from the audio files in test set

    Args:
        clean_json_file: Path to the clean json file containing all the test set clean audio files' paths as a list
        noisy_json_file: Path to the noise json file containing all the test set noise audio files' paths as a list
        fs: sampling frequency
        noisy_saving_path_base: path to the folder where generated noisy clips are saved
        clean_saving_path_base: path to the folder where corresponding clean clips are stored
        num_files: number of noisy files to be generated (default = None); if is None, then num_files = number of clean audio files

    '''
    if not os.path.isdir(noisy_saving_path_base):
        os.makedirs(noisy_saving_path_base)

    if not os.path.isdir(clean_saving_path_base):
        os.makedirs(clean_saving_path_base)

    with open(clean_json_file) as f:
        clean_files = json.load(f)

    with open(noise_json_file) as f:
        noise_files = json.load(f)
    
    if num_files == None:
        for i in range(len(clean_files)):
            noise_audio = random.choice(noise_files)
            generate_noisy_audio(clean_files[i], noise_audio, fs, os.path.join(noisy_saving_path_base, 'noisy') + str(i) +'.wav')
            shutil.copyfile(clean_files[i], os.path.join(clean_saving_path_base, 'clean') + str(i) +'.wav')
    else:
        random.shuffle(clean_files)
        for i in range(num_files):
            noise_audio = random.choice(noise_files)
            generate_noisy_audio(clean_files[i], noise_audio, fs, os.path.join(noisy_saving_path_base, 'noisy') + str(i) +'.wav')
            shutil.copyfile(clean_files[i], os.path.join(clean_saving_path_base, 'clean') + str(i) +'.wav')
    

def generate_noisy_audio(clean_file, noise_file, fs, path_for_saving):
    '''
    generate single noisy audio given clean and noise file

    Args:
        clean_file: path to the clean audio file
        noise_file: path to the noise audio file
        fs: sampling frequency
        path_for_saving: path to save the generated noisy file
    '''

    clean_audio, sr = read_audio(clean_file, fs, False)
    noise_signal, sr = read_audio(noise_file, fs, False)

    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    # Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    sf.write(path_for_saving, noisyAudio, int(sr))

def evaluate_test_audios(noisy_test_audio_path_base,clean_test_audio_path_base,model,fs,window_length,denoised_wav_path_base,csv_path,numFeatures,numSegments):
    '''
    run the given pytorch model on the given noisy clips, generated the denoised clips into given path, and generate the releveant testing results

    Args:
        noisy_test_aido_path_base: path to the folder of noisy clips
        clean_test_audio_path_base: path to the folder of corresponding clean clips
        model: path to the pytorch model
        fs: sampling frequency
        window_length: STFT window length
        denoised_wav_path_base: path to the folder for saving the denoised clips
        csv_path: path to the csv file for saving the per clip snr (input snr, output_snr, snr_improvement)
        numFeatures: number of frequency bins (feature) per second
        numSegments: number of consecutive timestemps used as model input
    '''

    if not os.path.isdir(denoised_wav_path_base):
        os.makedirs(denoised_wav_path_base)

    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = ['input_snr', 'output_snr', 'snr_improvement'])
    
    # save the noisy and clean's mean & std
    with open('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/denoised_audios/noisy_vs_clean_distribution.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = ['clean_mean', 'clean_std', 'noisy_mean', 'noisy_std'])

    pytorch_model = AudioDenoiserNet(numSegments)
    pytorch_model.load_state_dict(torch.load(model))

    pytorch_model.eval()

    noisy_test_files = sorted(os.listdir(noisy_test_audio_path_base))
    clean_test_files = sorted(os.listdir(clean_test_audio_path_base))
    assert len(noisy_test_files) == len(clean_test_files)

    for i in range(len(noisy_test_files)):
        denoised_wav_path = os.path.join(denoised_wav_path_base,'denoised_' + noisy_test_files[i])
        noisy_wav_file_path = os.path.join(noisy_test_audio_path_base,noisy_test_files[i])
        clean_wav_file_path = os.path.join(clean_test_audio_path_base,clean_test_files[i])
        evaluate_new_wav_signal(pytorch_model,noisy_wav_file_path,clean_wav_file_path,fs,window_length,denoised_wav_path,csv_path,numFeatures,numSegments)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Helpers for evaluating single noisy waveform file

def prepare_input_features_for_evaluation(stft_features, numFeatures, numSegments):
    '''
    transform the given stft spectrogram into model's input

    Args:

        numFeatures: number of frequency bins (feature) per second
        numSegments: number of consecutive timestemps used as model input
    '''

    noisySTFT = np.concatenate([stft_features[:, 0:numSegments - 1], stft_features], axis=1)
    stftSegments = np.zeros((noisySTFT.shape[1] - numSegments + 1,numFeatures, numSegments))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[index, :, :] = noisySTFT[:, index:index + numSegments]

    stftSegments = np.expand_dims(stftSegments, axis=1)
    return stftSegments


def revert_features_to_audio(noiseAudioFeatureExtractor,features,phase,cleanMean=None,cleanStd=None):
    '''
    revert the noisy stft back to waveform given

    The audio specific params (sampling rate, window_length, etc.) are defined in the noiseAudioFeatureExtractor class
    '''

    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    features = np.squeeze(features)
    features = np.transpose(features,(1,0))
    features = features * np.exp(1j * phase)

    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)


def evaluate_new_wav_signal(model,noisy_wav_file_path,clean_wav_file_path,fs,window_length,denoised_wav_path,csv_path,numFeatures,numSegments):
    '''
    given a pytorch model & noisy waveform, output the clean waveform

    Args:

        model: pytorch model
        wav_file_path: path to noisy waveform
        fs: signal sampling rate in Hz
        window_length: STFT window length
        denoised_wav_path: path to save the predicted clean waveform
        csv_path: path to save the input and output snr as csv file
        numFeatures: number of frequency bins (feature) per second
        numSegments: number of consecutive timestemps used as model input

    '''
    overlap = round(0.25 * window_length) # overlap of 75%
    noisyAudio, sr = read_audio(noisy_wav_file_path, fs, False)
    cleanAudio, sr = read_audio(clean_wav_file_path, fs, False)
    input_snr = calculate_input_snr(cleanAudio, noisyAudio, False)
    noiseAudioFeatureExtractor = FeatureExtractor(noisyAudio, windowLength=window_length, overlap=overlap, sample_rate=sr)
    noisy_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    noisyPhase = np.angle(noisy_stft_features)
    cleanAudioFeatureExtractor = FeatureExtractor(cleanAudio, windowLength=window_length, overlap=overlap, sample_rate=sr)
    clean_stft_features = cleanAudioFeatureExtractor.get_stft_spectrogram()
    # cleanPhase = np.angle(clean_stft_features)

    noisy_stft_features = np.abs(noisy_stft_features)
    mean = np.mean(noisy_stft_features)
    std = np.std(noisy_stft_features)
    noisy_stft_features = (noisy_stft_features - mean) / std

    clean_stft_features = np.abs(clean_stft_features)
    clean_mean = np.mean(clean_stft_features)
    clean_std = np.std(clean_stft_features)
    clean_stft_features = (clean_stft_features - clean_mean) / clean_std

    input = prepare_input_features_for_evaluation(noisy_stft_features,numFeatures,numSegments)
    input = torch.from_numpy(input).float()

    prediction = model(input).detach().numpy()
    denoisedWaveform = revert_features_to_audio(noiseAudioFeatureExtractor, prediction, noisyPhase)
    output_snr = calculate_output_snr(cleanAudio, denoisedWaveform, False)
    snr_improvement = output_snr - input_snr
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([input_snr, output_snr, snr_improvement])

    # save the noisy and clean's mean & std
    with open('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/denoised_audios/noisy_vs_clean_distribution.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([clean_mean, clean_std, mean, std])

    sf.write(denoised_wav_path, denoisedWaveform, int(sr))


# -----------------------------------------------------------------------------------------------------------------------------------
# Helpers for quantitive metrics
def calculate_input_noise_sum(clean_signal, input_signal, gpu=True):
    if gpu:
        clean_signal = clean_signal.cpu().data.numpy()
        input_signal = input_signal.cpu().data.numpy()
    input_noise = input_signal - clean_signal

    return np.sum(np.power(input_noise, 2))

def calculate_output_noise_sum(clean_signal, output_signal, gpu=True):
    if gpu:
        clean_signal = clean_signal.cpu().data.numpy()
        output_signal = output_signal.cpu().data.numpy()
    output_noise = output_signal - clean_signal

    return np.sum(np.power(output_noise, 2))

def calculate_input_snr(clean_signal, input_signal, gpu=True):
    if gpu:
        clean_signal = clean_signal.cpu().data.numpy()
        input_signal = input_signal.cpu().data.numpy()

    noise_signal = input_signal - clean_signal

    clean_signal_squared = np.power(clean_signal, 2)
    noise_signal_squared = np.power(noise_signal, 2)

    input_snr = (np.sum(clean_signal_squared))/(np.sum(noise_signal_squared))

    return 10*(math.log10(input_snr))

def calculate_output_snr(clean_signal, output_signal, gpu=True):
    if gpu:
        clean_signal = clean_signal.cpu().data.numpy()
        output_signal = output_signal.cpu().data.numpy()

    noise_signal = output_signal - clean_signal

    clean_signal_squared = np.power(clean_signal, 2)
    noise_signal_squared = np.power(noise_signal, 2)

    output_snr = (np.sum(clean_signal_squared))/(np.sum(noise_signal_squared))

    return 10*(math.log10(output_snr))

# -----------------------------------------------------------------------------------------------------------------------------------
# Class and Helpers for Plotting

class csv_generator():
    def __init__(self,loss_per_epoch_csv,loss_per_batch_csv,input_output_snr_difference_csv,log_ratio_output_input_noise_csv):
        self.loss_per_epoch = loss_per_epoch_csv
        self.loss_per_batch = loss_per_batch_csv
        self.input_output_snr_difference = input_output_snr_difference_csv
        self.log_ratio_output_input_noise = log_ratio_output_input_noise_csv
        # Loss per epoch
        with open(self.loss_per_epoch, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = ['epoch_number', 'loss'])

        # Loss per batch
        with open(self.loss_per_batch, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = ['epoch_number', 'batch_number', 'loss'])

        # Input Output SNR difference
        with open(self.input_output_snr_difference, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = ['epoch_number', 'batch_number', 'input_snr', 'output_snr'])

        # Log ratio Output Input noise
        with open(self.log_ratio_output_input_noise, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames = ['epoch_number', 'batch_number', 'input_noise', 'output_noise', 'log_ratio']) 

    def save_loss_per_epoch(self, epoch_number, loss):
        with open(self.loss_per_epoch, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_number, loss])

    def save_loss_per_batch(self, epoch_number, batch_number, loss):
        with open(self.loss_per_batch, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_number, batch_number, loss])

    def save_input_output_snr_difference(self, epoch_number, batch_number, input_snr, output_snr):
        with open(self.input_output_snr_difference, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_number, batch_number, input_snr, output_snr])

    def save_log_ratio_output_input_noise(self, epoch_number, batch_number, log_ratio):
        with open(self.log_ratio_output_input_noise, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_number, batch_number, log_ratio])


def plot_loss(training_loss,valid_loss,saving_path):
    '''
    plot training loss and validation loss on the same plot
    '''
    plt.figure()
    epochs = np.arange(1,len(training_loss)+1)
    plt.plot(epochs,training_loss,label='Training Curve')
    plt.plot(epochs,valid_loss,label='Validation Curve')
    plt.title("Training Loss vs. Validation Loss")
    plt.ylabel('Loss')
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(saving_path+"loss.png")

def plot_spectrogram(S, name, saving_path):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.savefig(saving_path + name +"_spectrogram.png")
