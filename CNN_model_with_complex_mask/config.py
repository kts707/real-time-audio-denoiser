# Process Raw Dataset

# Specify the raw dataset location to get the clean and noise signal files from the dataset
CLEAN_BASEPATH = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/raw_path/clips"
NOISY_BASEPATH = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/raw_path/noise"

# Specify the path for saving the preprocessed numpy array data
PREPROCESSED_CLEAN_TARGET = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/dataset/clean"
PREPROCESSED_NOISY_INPUT = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/dataset/noisy"

# json files for storing test audio file names
CLEAN_JSON_FILE = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/clean_test.json'
NOISE_JSON_FILE = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/noise_test.json'

# ------------------------------------------------------------------------------------------------------------------------
# Training Parameters

# Processed Data Directory
DATA_DIR = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/dataset"

# Directory for saving training curve and other results
RESULTS = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/results/"

# log directory for tensorboard visualization
LOG_DIR = "/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/results/log_dir"

# sequence length used in the model
SEQUENCE_LENGTH = 8

# Training Hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

# ---------------------------------------------------------------------------------------------------------------------------
# Test Parameters
# params for test.py

# Change these only when dataset changed
CLEAN_JSON_FILE_TEST = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/clean_test.json'
NOISE_JSON_FILE_TEST = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/noise_test.json'
# NOISY_TEST_FILE_SAVING_PATH = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/noisy_test_audios'
# CLEAN_TEST_FILE_SAVING_PATH = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/clean_test_audios'
NOISY_TEST_FILE_SAVING_PATH = '/local/mnt2/workspace2/tkuai/datasets/raw/noisy_testset_wav'
CLEAN_TEST_FILE_SAVING_PATH = '/local/mnt2/workspace2/tkuai/datasets/raw/clean_testset_wav'



# Change these for testing a new model
DENOISED_TEST_FILE_SAVING_PATH = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/valentini_test/denoised_wav'
TEST_STATS = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/valentini_test/test_stats/per_clip_stats.csv'
TEST_MODEL = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/model1/AudioDenoiser_512_0.001_49.pt'

DUMPED_ONNX_MODEL = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/model2/AudioDenoiser_512_0.001_49.onnx'

fs = 16000
num_files = 10
window_length = 256
overlap      = round(0.25 * window_length) # overlap of 75%
ffTLength    = window_length
inputFs      = 48e3
fs           = 16e3
numFeatures  = ffTLength//2 + 1
numSegments  = 8


