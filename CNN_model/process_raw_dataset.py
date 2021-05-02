import numpy as np
import librosa
import torch
import scipy
import os
import math
from sklearn.preprocessing import StandardScaler
import warnings
import json
import soundfile as sf
import random
import utils

class FeatureExtractor:
    def __init__(self, audio, *, windowLength, overlap, sample_rate):
        self.audio = audio
        self.ffT_length = windowLength
        self.window_length = windowLength
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window = scipy.signal.hamming(self.window_length, sym=False)

    def get_stft_spectrogram(self):
        return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            window=self.window, center=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap,
                             window=self.window, center=True)

class CleanDataset:
    def __init__(self, basepath):
        self.basepath = basepath

    def _get_clean_filenames(self):
        clean_files = os.listdir(self.basepath)
        print("Total number of clean files:", len(clean_files))
        return clean_files

    def get_train_validation_test_filenames(self):
        clean_files = self._get_clean_filenames()
        np.random.shuffle(clean_files)
        # resolve full path
        clean_files = [os.path.join(self.basepath, filename) for filename in clean_files]
        total = len(clean_files)

        # Train:Validation:Test = 0.7:0.15:0.15
        clean_train_files = clean_files[:total-int(0.3*total)]
        clean_val_files = clean_files[-int(0.3*total):-int(0.15*total)]
        clean_test_files = clean_files[-int(0.15*total):]
        print("# of clean Training files:", len(clean_files))
        print("# of clean Validation files:", len(clean_val_files))
        print("# of clean Test files:",len(clean_test_files))
        return clean_train_files, clean_val_files, clean_test_files

class NoiseDataset:
    def __init__(self, basepath):
        self.basepath = basepath

    def _get_noise_filenames(self):
        noise_files = os.listdir(self.basepath)
        print("Total number of noise files:", len(noise_files))
        return noise_files

    def get_train_validation_test_filenames(self):
        noise_files = self._get_noise_filenames()
        noise_filenames = [os.path.join(self.basepath, filename) for filename in noise_files]
        np.random.shuffle(noise_filenames)

        total = len(noise_files)
        # separate noise files for train/validation
        noise_train_files = noise_filenames[:total-int(0.3*total)]
        noise_val_files = noise_filenames[-int(0.3*total):-int(0.15*total)]
        noise_test_files = noise_filenames[-int(0.15*total):]
        print("# of noise training files:", len(noise_train_files))
        print("# of noise validation files:", len(noise_val_files))
        print("# of noise test files:", len(noise_test_files))

        return noise_train_files, noise_val_files, noise_test_files


class Dataset:
    def __init__(self, clean_filenames, noise_filenames, base_saving_path_for_clean_target, base_saving_path_for_noisy_inputs,**config):
        self.clean_filenames = clean_filenames
        self.noise_filenames = noise_filenames
        self.saving_path_for_clean_target = base_saving_path_for_clean_target
        self.saving_path_for_noisy_inputs = base_saving_path_for_noisy_inputs
        self.sample_rate = config['fs']
        self.overlap = config['overlap']
        self.window_length = config['windowLength']
        self.audio_max_duration = config['audio_max_duration']
        self.noise_scale_factor = config['noise_scale_factor']
        self.phase_aware_scaling = config['apply_phase_aware_scaling']
        self.normalize_to_standard_normal_distribution = config['normalize_to_standard_normal_distribution']

    def _sample_noise_filename(self):
        return np.random.choice(self.noise_filenames)

    def _remove_silent_frames(self, audio):
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=self.overlap, top_db=20)

        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

    def _phase_aware_scaling(self, clean_spectral_magnitude, clean_phase, noise_phase):
        assert clean_phase.shape == noise_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)

    def get_noisy_audio(self, *, filename):
        return read_audio(filename, self.sample_rate, False)

    def _audio_random_crop(self, audio, duration):
        audio_duration_secs = librosa.core.get_duration(audio, self.sample_rate)

        ## duration: length of the cropped audio in seconds
        if duration >= audio_duration_secs:
            print("Passed duration greater than audio duration of: ", audio_duration_secs)
            return audio

        audio_duration_ms = math.floor(audio_duration_secs * self.sample_rate)
        duration_ms = math.floor(duration * self.sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return audio[idx: idx + duration_ms]

    def _add_noise_to_clean_audio(self, clean_audio, noise_signal):
        if len(clean_audio) >= len(noise_signal):
            # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
            while len(clean_audio) >= len(noise_signal):
                noise_signal = np.append(noise_signal, noise_signal)

        ## Extract a noise segment from a random location in the noise file
        ind = np.random.randint(0, noise_signal.size - clean_audio.size)

        noiseSegment = noise_signal[ind: ind + clean_audio.size]

        speech_power = np.sum(clean_audio ** 2)
        noise_power = np.sum(noiseSegment ** 2)
        noisyAudio = clean_audio + self.noise_scale_factor * np.sqrt(speech_power / noise_power) * noiseSegment
        return noisyAudio

    def audio_processing(self, clean_filename):
        # read the clean audio
        clean_audio, _ = read_audio(clean_filename, self.sample_rate, False)

        # remove silent frame from clean audio
        clean_audio = self._remove_silent_frames(clean_audio)

        noise_filename = self._sample_noise_filename()

        # read the noise file
        noise_audio, sr = read_audio(noise_filename, self.sample_rate, False)

        # remove silent frame from noise audio
        noise_audio = self._remove_silent_frames(noise_audio)

        # sample random fixed-sized snippets of audio
        clean_audio = self._audio_random_crop(clean_audio, duration=self.audio_max_duration)

        # add noise to clean audios
        noiseInput = self._add_noise_to_clean_audio(clean_audio, noise_audio)
        # a = str(random.randint(0,10))
        # sf.write('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/temp/test_' + a + '.wav', noiseInput, int(sr))

        # extract stft features from noisy audio
        noisy_input_feature = FeatureExtractor(noiseInput, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        noisy_spectrogram = noisy_input_feature.get_stft_spectrogram()

        # get the magnitude of the spectral & normalize

        noisy_magnitude = np.abs(noisy_spectrogram)
        # noisy_mean = np.mean(noisy_magnitude)
        # noisy_std = np.std(noisy_magnitude)
        # noisy_magnitude = (noisy_magnitude - noisy_mean) / noisy_std

        # extract stft features from clean audio
        clean_audio_fe = FeatureExtractor(clean_audio, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        clean_spectrogram = clean_audio_fe.get_stft_spectrogram()


        # # get the clean spectral magnitude
        clean_magnitude = np.abs(clean_spectrogram)
        # clean_mean = np.mean(clean_magnitude)
        # clean_std = np.std(clean_magnitude)
        # clean_magnitude = (clean_magnitude - clean_mean) / clean_std

        if self.phase_aware_scaling:
            # get the noisy phase and clean phase
            noisy_phase = np.angle(noisy_spectrogram)
            clean_phase = np.angle(clean_spectrogram)
            clean_magnitude = self._phase_aware_scaling(clean_magnitude, clean_phase, noisy_phase)

        if self.normalize_to_standard_normal_distribution:
            scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
            noisy_magnitude = scaler.fit_transform(noisy_magnitude)
            clean_magnitude = scaler.transform(clean_magnitude)

        # noisy_recover = revert_features_to_audio(noisy_input_feature, noisy_magnitude, noisy_phase, noisy_mean, noisy_std)
        # clean_recover = revert_features_to_audio(clean_audio_fe, clean_spectrogram, clean_phase, np.mean(clean_spectrogram), np.std(clean_spectrogram))
        # noisy_recover = revert_features_to_audio(noisy_input_feature, noisy_spectrogram, noisy_phase)
        # clean_recover = revert_features_to_audio(clean_audio_fe, clean_magnitude, clean_phase)
        # sf.write('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/temp/noisy_recovered_' + a + '.wav', noisy_recover, int(sr))
        # sf.write('/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/temp/clean_recovered' + a + '.wav', clean_recover, int(sr))
        # print("noisy stft shape = ",noisy_magnitude.shape)
        # print("clean magnitude =",clean_magnitude.shape)
        return noisy_magnitude, clean_magnitude

    def preprocess_raw_audio_files(self, *, prefix, subset_size):
        counter = 0

        for i in range(0, len(self.clean_filenames), subset_size):
            clean_filenames_sublist = self.clean_filenames[i:i + subset_size]
            out = [self.audio_processing(filename) for filename in clean_filenames_sublist]
            k = 0
            for o in out:
                k += 1
                noisy_stft_magnitude = o[0]
                clean_stft_magnitude = o[1]

                np.save(self.saving_path_for_clean_target + '/' + prefix + str(counter) + '_' + str(k) + '.npy',clean_stft_magnitude)
                np.save(self.saving_path_for_noisy_inputs + '/' + prefix + str(counter) + '_' + str(k) + '.npy',noisy_stft_magnitude)

            counter += 1


def read_audio(filepath, sample_rate, normalize=True):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize is True:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
        # audio = librosa.util.normalize(audio)
    return audio, sr


def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # If the noise signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio


if __name__ == '__main__':
    # Create Dataset

    # Specify the raw dataset location to get the clean and noise signal files from the dataset
    # This is the Urbansound8K + Mozilla Dataset
    clean_basepath = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/raw_path/clips"
    noise_basepath = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/raw_path/noise"

    np.random.seed(14)
    warnings.filterwarnings(action='ignore')

    clean_data = CleanDataset(clean_basepath)
    clean_train_filenames, clean_val_filenames, clean_test_filenames = clean_data.get_train_validation_test_filenames()
    noise_data = NoiseDataset(noise_basepath)
    noise_train_filenames, noise_val_filenames, noise_test_filenames = noise_data.get_train_validation_test_filenames()

    # json files for storing test audio file names
    clean_json_file = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/clean_test2.json'
    noise_json_file = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/noise_test2.json'

    with open(noise_json_file,'w') as f:
        json.dump(noise_test_filenames, f)

    with open(clean_json_file,'w') as f:
        json.dump(clean_test_filenames, f)

    # preprocessing audio files
    windowLength = 256
    config = {'windowLength': windowLength,
              'overlap': round(0.25 * windowLength),
              'fs': 16000,
              'audio_max_duration': 0.75,
              'noise_scale_factor': 1,
              'apply_phase_aware_scaling': False,
              'normalize_to_standard_normal_distribution': False}

    # Specify the base path for saving the preprocessed clean target and noisy inputs

    clean_target_base_path_for_saving = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_dataset2/old_data/clean"
    noisy_input_base_path_for_saving = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_dataset2/old_data/noisy"

    # create the directories if they do not exist
    if not os.path.isdir(clean_target_base_path_for_saving):
        os.makedirs(clean_target_base_path_for_saving)
        os.mkdir(os.path.join(clean_target_base_path_for_saving,'training'))
        os.mkdir(os.path.join(clean_target_base_path_for_saving,'validation'))
        # os.mkdir(os.path.join(clean_target_base_path_for_saving,'test'))

    if not os.path.isdir(noisy_input_base_path_for_saving):
        os.makedirs(noisy_input_base_path_for_saving)
        os.mkdir(os.path.join(noisy_input_base_path_for_saving,'training'))
        os.mkdir(os.path.join(noisy_input_base_path_for_saving,'validation'))
        # os.mkdir(os.path.join(noisy_input_base_path_for_saving,'test'))

    clean_training_path = os.path.join(clean_target_base_path_for_saving,'training')
    clean_validation_path = os.path.join(clean_target_base_path_for_saving,'validation')
    # clean_test_path = os.path.join(clean_target_base_path_for_saving,'test')
    noisy_training_path = os.path.join(noisy_input_base_path_for_saving,'training')
    noisy_validation_path = os.path.join(noisy_input_base_path_for_saving,'validation')
    # noisy_test_path = os.path.join(noisy_input_base_path_for_saving,'test')


    print("start processing data for training set")
    train_dataset = Dataset(clean_train_filenames, noise_train_filenames, clean_training_path, noisy_training_path, **config)
    train_dataset.preprocess_raw_audio_files(prefix = "training", subset_size=400)

    print("start processing data for validation set")
    val_dataset = Dataset(clean_val_filenames, noise_val_filenames, clean_validation_path, noisy_validation_path, **config)
    val_dataset.preprocess_raw_audio_files(prefix = "validation", subset_size=400)


    # disable processing for test set, the test set file paths are saved above in the json files

    # print("start processing data for test set")
    # test_dataset = Dataset(clean_test_filenames, noise_test_filenames, clean_test_path, noisy_test_path, **config)
    # test_dataset.preprocess_raw_audio_files(prefix = "test", subset_size=400)