from utils import *
import warnings
import config as cfg
import onnx


# Change these only when dataset changed
clean_json_file = cfg.CLEAN_JSON_FILE_TEST
noise_json_file = cfg.NOISE_JSON_FILE_TEST
noisy_saving_path_base = cfg.NOISY_TEST_FILE_SAVING_PATH
clean_saving_path_base = cfg.CLEAN_TEST_FILE_SAVING_PATH


# Change these for testing a new model
denoised_audio_path = cfg.DENOISED_TEST_FILE_SAVING_PATH
csv_path = cfg.TEST_STATS
model = cfg.TEST_MODEL
dumped_onnx = cfg.DUMPED_ONNX_MODEL


warnings.filterwarnings('ignore')
# generate_noisy_waveform_for_test_set(clean_json_file, noise_json_file, fs, noisy_saving_path_base, clean_saving_path_base)
evaluate_test_audios(noisy_saving_path_base,clean_saving_path_base,model,cfg.fs,cfg.window_length,denoised_audio_path,csv_path,cfg.numFeatures,cfg.numSegments)

# dump_onnx_model(model,[1,2,cfg.numFeatures,cfg.numSegments],dumped_onnx)