import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import math
import csv
import os
import sys
import random
import json
import warnings
import shutil
from tqdm import tqdm
from pypesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz
import onnx

from process_raw_dataset import FeatureExtractor, read_audio
from model import AudioDenoiserNet
import config as cfg

# ------------------------------------------------------------------------------------------------------------------------------------------
# Helper functions for generating training data from noisy spectrogram
def prepare_input_features(noisy_stft_features, clean_stft_features, numSegments, numFeatures):
    '''
    This function prepares the input features for training

    Args:

    numSegments = number of time stamp per input sample

    numFeatures = number of frequency bins for each time stamp
    '''

    stftSegments = np.zeros((noisy_stft_features.shape[2] - numSegments + 1, 2, numFeatures, numSegments))

    for index in range(stftSegments.shape[0]):
        stftSegments[index, :, :, :] = noisy_stft_features[:, :, index:index + numSegments]

    clean_stft_features = clean_stft_features[:,:,7:]

    clean_stft_features = np.transpose(clean_stft_features, (2, 0, 1))
    clean_stft_features = np.expand_dims(clean_stft_features, axis=[3])

    return stftSegments, clean_stft_features

# ----------------------------------------------------------------------------------------------------------------------------------------
# Helpers for getting qualitative results for test set given json files of clean audio and noisy audio

def generate_noisy_waveform_for_test_set(clean_json_file, noise_json_file, fs, noisy_saving_path_base, clean_saving_path_base, num_files = None):

    if not os.path.isdir(noisy_saving_path_base):
        os.makedirs(noisy_saving_path_base)

    if not os.path.isdir(clean_saving_path_base):
        os.makedirs(clean_saving_path_base)

    with open(clean_json_file) as f:
        clean_files = json.load(f)

    with open(noise_json_file) as f:
        noise_files = json.load(f)
    
    print("total number of clean files = ",len(clean_files))

    if num_files == None:
        print("generate",len(clean_files),"noisy test files")
        for i in range(len(clean_files)):
            noise_audio = random.choice(noise_files)
            generate_noisy_audio(clean_files[i], noise_audio, fs, os.path.join(noisy_saving_path_base, 'noisy') + str(i) +'.wav')
            shutil.copyfile(clean_files[i], os.path.join(clean_saving_path_base, 'clean') + str(i) +'.wav')
    else:
        random.shuffle(clean_files)
        print("generate",num_files,"noisy test files")
        for i in range(num_files):
            noise_audio = random.choice(noise_files)
            generate_noisy_audio(clean_files[i], noise_audio, fs, os.path.join(noisy_saving_path_base, 'noisy') + str(i) +'.wav')
            shutil.copyfile(clean_files[i], os.path.join(clean_saving_path_base, 'clean') + str(i) +'.wav')
    

def generate_noisy_audio(clean_file, noise_file, fs, path_for_saving):
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
    if not os.path.isdir(denoised_wav_path_base):
        os.makedirs(denoised_wav_path_base)

    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = ['noisy_file_name','noisy_input_csig','noisy_input_cbak','noisy_input_covl','noisy_input_pesq','noisy_input_segSNR','noisy_input_stoi',\
            'csig','cbak','covl','pesq','segSNR','stoi','input_snr','output_snr','snr_improvement','csig_improvement','cbak_improvement','covl_improvement',\
                'pesq_improvement','segSNR_improvement','stoi_improvement'])
        writer.writeheader()

        # writer.writerow(['noisy_file_name','noisy_input_csig','noisy_input_cbak','noisy_input_covl','noisy_input_pesq','noisy_input_segSNR','noisy_input_stoi',\
        #     'csig','cbak','covl','pesq','segSNR','stoi','input_snr','output_snr','snr_improvement'])
    
    # with open('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/denoised_audios/noisy_vs_clean_distribution.csv', 'w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames = ['clean_mean', 'clean_std', 'noisy_mean', 'noisy_std'])

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
        noisy_input_stats, denoised_stats = evaluate_new_wav_signal(pytorch_model,noisy_wav_file_path,clean_wav_file_path,fs,window_length,denoised_wav_path,csv_path,numFeatures,numSegments)

        # noisy input stats
        noisy_csig = noisy_input_stats['csig']
        noisy_cbak = noisy_input_stats['cbak']
        noisy_covl = noisy_input_stats['covl']
        noisy_pesq = noisy_input_stats['pesq']
        noisy_segSNR = noisy_input_stats['segSNR']
        noisy_stoi = noisy_input_stats['stoi']

        # denoised stats
        csig = denoised_stats['csig']
        cbak = denoised_stats['cbak']
        covl = denoised_stats['covl']
        pesq = denoised_stats['pesq']
        segSNR = denoised_stats['segSNR']
        stoi = denoised_stats['stoi']
        input_snr = denoised_stats['input_snr']
        output_snr = denoised_stats['output_snr']
        snr_improvement = denoised_stats['snr_improvement']

        # write to csv file
        with open(csv_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([noisy_test_files[i],noisy_csig,noisy_cbak,noisy_covl,noisy_pesq,noisy_segSNR,noisy_stoi,\
            csig,cbak,covl,pesq,segSNR,stoi,input_snr,output_snr,snr_improvement,csig-noisy_csig,cbak-noisy_cbak,covl-noisy_covl,\
                pesq-noisy_pesq,segSNR-noisy_segSNR,stoi-noisy_stoi])

# ----------------------------------------------------------------------------------------------------------------------------------------
# Helpers for evaluating single noisy waveform file

def prepare_input_features_for_evaluation(stft_features, numFeatures, numSegments):
    '''
    transform the given stft spectrogram into model's input

    Args:

    numFeatures: number of frequency bins (feature) per second

    numSegments: number of consecutive timestemps used as model input
    '''

    noisySTFT = np.concatenate([stft_features[:, :, 0:numSegments - 1], stft_features], axis=2)
    stftSegments = np.zeros((noisySTFT.shape[2] - numSegments + 1, 2, numFeatures, numSegments))

    for index in range(noisySTFT.shape[2] - numSegments + 1):
        stftSegments[index, :, :, :] = noisySTFT[:, :, index:index + numSegments]

    return stftSegments


def revert_features_to_audio(noisyAudioFeatureExtractor,features,cleanMean=None,cleanStd=None):
    '''
    revert the noisy stft back to waveform given

    The audio specific params (sampling rate, window_length, etc.) are defined in the noiseAudioFeatureExtractor class
    '''

    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean
    # print('feature shape = ',features.shape)

    assert features.shape[1] == 2, "must have two channels -> real & imaginary"

    output_spectrogram = features[:,0,:,:] + 1j*features[:,1,:,:]
    # print('combine channels shape = ',output_spectrogram.shape)
    output_spectrogram = np.squeeze(output_spectrogram)
    output_spectrogram = np.transpose(output_spectrogram,(1,0))
    return noisyAudioFeatureExtractor.get_audio_from_stft_spectrogram(output_spectrogram)


def evaluate_new_wav_signal(model,noisy_wav_file_path,clean_wav_file_path,fs,window_length,denoised_wav_path,csv_path,numFeatures,numSegments):
    '''
    given a pytorch model & noisy waveform, output the clean waveform

    Args:

    model: pytorch model

    wav_file_path: path to noisy waveform

    fs: signal sampling rate in Hz

    window_length = STFT window length

    denoised_wav_path: path to save the predicted clean waveform

    csv_path: path to save the input and output snr as csv file

    numFeatures: number of frequency bins (feature) per second

    numSegments: number of consecutive timestemps used as model input

    '''
    overlap = round(0.25 * window_length) # overlap of 75%
    noisyAudio, sr = read_audio(noisy_wav_file_path, fs, False)
    cleanAudio, sr = read_audio(clean_wav_file_path, fs, False)

    noiseAudioFeatureExtractor = FeatureExtractor(noisyAudio, windowLength=window_length, overlap=overlap, sample_rate=sr)
    noisy_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    noisy_input = np.stack((noisy_stft_features.real,noisy_stft_features.imag))

    # noisyPhase = np.angle(noisy_stft_features)
    # cleanAudioFeatureExtractor = FeatureExtractor(cleanAudio, windowLength=window_length, overlap=overlap, sample_rate=sr)
    # clean_stft_features = cleanAudioFeatureExtractor.get_stft_spectrogram()

    # clean_target = np.stack((clean_stft_features.real,clean_stft_features.imag))

    # cleanPhase = np.angle(clean_stft_features)

    # noisy_stft_features = np.abs(noisy_stft_features)
    # mean = np.mean(noisy_stft_features)
    # std = np.std(noisy_stft_features)
    # noisy_stft_features = (noisy_stft_features - mean) / std

    # clean_stft_features = np.abs(clean_stft_features)
    # clean_mean = np.mean(clean_stft_features)
    # clean_std = np.std(clean_stft_features)
    # clean_stft_features = (clean_stft_features - clean_mean) / clean_std

    model_input = prepare_input_features_for_evaluation(noisy_input,numFeatures,numSegments)

    model_input = torch.from_numpy(model_input).float()

    prediction = model(model_input).detach().numpy()
    denoisedWaveform = revert_features_to_audio(noiseAudioFeatureExtractor, prediction)

    # calculate
    input_snr = calculate_input_snr(cleanAudio, noisyAudio, False)
    output_snr = calculate_output_snr(cleanAudio, denoisedWaveform, False)

    snr_improvement = output_snr - input_snr

    noisy_input_stats = eval_composite(noisyAudio,cleanAudio,int(sr))
    denoised_stats = eval_composite(denoisedWaveform,cleanAudio,int(sr))

    denoised_stats['input_snr'] = input_snr
    denoised_stats['output_snr'] = output_snr
    denoised_stats['snr_improvement'] = snr_improvement

    # with open('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/denoised_audios/noisy_vs_clean_distribution.csv', 'a') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([clean_mean, clean_std, mean, std])

    sf.write(denoised_wav_path, denoisedWaveform, int(sr))
    return noisy_input_stats, denoised_stats


# -----------------------------------------------------------------------------------------------------------------------------------
# quantitive metrics

'''
    Currently support:
        SNR: Signal to Noise Ratio
        SegSNR: Segmental SNR
        CSIG: a prediction of the signal distortion attending only to the speech signal
        CBAK: a prediction of the intrusiveness of background noise
        COVL: a prediction of the overall denoising quality
        PESQ: Perceptual Evaluation of Speech Quality
        STOI: Short Term Objective Intelligibility Measures


'''


def eval_composite(output_wav, target_wav, sr = 16000):
    '''
    This function evaluate the denoising based on the metrics listed above

    Args:
        output_wav: model output waveform
        target_wav: clean target waveform
        sr: sampling rate (must be int)
    '''

    target_wav = target_wav.reshape(-1)
    output_wav = output_wav.reshape(-1)

    alpha = 0.95
    len_ = min(target_wav.shape[0], output_wav.shape[0])
    target_wav = target_wav[:len_]

    output_wav = output_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, output_wav, sr)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, output_wav, sr)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, output_wav, sr)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = calculate_pesq(output_wav, target_wav, sr)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)

    stoi = calculate_stoi(output_wav, target_wav, sr)

    return {'csig':Csig, 'cbak':Cbak, 'covl':Covl, 'pesq':pesq_raw, 'segSNR':segSNR, 'stoi':stoi}


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

    len_ = min(clean_signal.shape[0], output_signal.shape[0])
    clean_signal = clean_signal[:len_]
    output_signal = output_signal[:len_]

    noise_signal = output_signal - clean_signal

    clean_signal_squared = np.power(clean_signal, 2)
    noise_signal_squared = np.power(noise_signal, 2)

    output_snr = (np.sum(clean_signal_squared))/(np.sum(noise_signal_squared))

    return 10*(math.log10(output_snr))

def SSNR(target_wav, output_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = target_wav
    processed_speech = output_wav
    clean_length = target_wav.shape[0]
    processed_length = output_wav.shape[0]
    
    # scale both to have same dynamic range. Remove DC too.
    clean_speech -= clean_speech.mean()
    processed_speech -= processed_speech.mean()
    processed_speech *= (np.max(np.abs(clean_speech)) / np.max(np.abs(processed_speech)))
   
    # Signal-to-Noise Ratio 
    dif = target_wav - output_wav
    overall_snr = 10 * np.log10(np.sum(target_wav ** 2) / (np.sum(dif ** 2) +
                                                        10e-20))


    # global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs


    skiprate = winlength // 4

    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength/skiprate))

    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr

def calculate_pesq(output, target, sr=16000):
    # pesq
    return pesq(target, output, sr)

def calculate_stoi(output, target, sr=16000, extended=False):
    # stoi
    return stoi(target, output, sr, extended=extended)

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

def trim_mos(val):
    return min(max(val, 1), 5)

def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocor lags
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))

    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i+1] - sum_term)/E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr = np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)

    return acorr, refcoeff, lpparams

def wss(target_wav, output_wav, srate):
    clean_speech = target_wav
    processed_speech = output_wav
    clean_length = target_wav.shape[0]
    processed_length = output_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25 # num of critical bands

    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
    n_fftby2 = int(n_fft / 2)

    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions (Center frequency and BW in Hz)
    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30, 
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423, 
                 153.823, 168.154, 183.457, 199.776, 217.153, 
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0] # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.
    min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
                                   norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
                                                 min_factor)

    # For each frame of input speech, compute Weighted Spectral Slope Measure
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0 # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed
        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit

        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
                                     crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
                                         crit_filter[i, :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))

        # (4) Compute Spectral Shape (dB[i+1] - dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - \
                processed_energy[:num_crit-1]

        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])

        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)

        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral 
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
                                   clean_energy[:num_crit-1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - \
                                processed_energy[:num_crit-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
                                      processed_energy[:num_crit-1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - \
                                     processed_slope[:num_crit - 1]) ** 2))

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion


def llr(target_wav, output_wav, srate):
    clean_speech = target_wav
    processed_speech = output_wav
    clean_length = target_wav.shape[0]
    processed_length = output_wav.shape[0]
    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]

        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)

        if (numerator/denominator) <= 0:
            print(f'Numerator: {numerator}')
            print(f'Denominator: {denominator}')

        log_ = np.log(numerator / denominator)
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.nan_to_num(np.array(distortion))

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


# -------------------------------------------------------------------------------------------------------------------------
# onnx related helpers

def load_state_dict(torch_model_state_dict):
    torch_model = AudioDenoiserNet(cfg.numSegments)
    torch_model.load_state_dict(torch.load(torch_model_state_dict))
    torch_model.eval()
    return torch_model

def dump_onnx_model(torch_model_state_dict, dummy_input_dim, saving_path):

    torch_model = AudioDenoiserNet(cfg.numSegments)
    torch_model.load_state_dict(torch.load(torch_model_state_dict))
    torch_model.eval()

    dummy_input = torch.randn(dummy_input_dim)
    torch.onnx.export(torch_model, dummy_input, saving_path)
