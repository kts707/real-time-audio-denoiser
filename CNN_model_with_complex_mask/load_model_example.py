import utils
import torch

torch_state_dict_path = '/local/mnt2/workspace2/tkuai/cnn_audio_denoiser/pytorch_model3/model2/AudioDenoiser_512_0.001_49.pt'
model = utils.load_state_dict(torch_state_dict_path)