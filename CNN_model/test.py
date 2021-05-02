from utils import *
import warnings

# json files for storing the names of the clean and noise files in test set
clean_json_file = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/clean_test.json'
noise_json_file = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/noise_test.json'

# paths for storing the generated noisy files, denoised files, and the pytorch model
saving_path_base = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/test_audios'
denoised_audio_path = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/denoised_audios'
model = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model2/AudioDenoiser_64_0.001_18.pt'


# params
fs = 16000
num_files = 10
window_length = 256
overlap      = round(0.25 * window_length) # overlap of 75%
ffTLength    = window_length
inputFs      = 48e3
fs           = 16e3
numFeatures  = ffTLength//2 + 1
numSegments  = 8

warnings.filterwarnings('ignore')

# generate 10 test noisy audio files from test set
generate_noisy_waveform_for_test_set(clean_json_file, noise_json_file, fs, saving_path_base, 10)

# evaluate the given pytorch model on the 10 noisy audio generated
evaluate_test_audios(saving_path_base,model,fs,window_length,denoised_audio_path,numFeatures,numSegments)
