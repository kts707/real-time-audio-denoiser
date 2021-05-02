# audio_denoiser
Real-Time CNN based Audio Denoiser

## Environmen Setup

Create the conda environment from the `environment.yml` file:

~~~
conda env create -f environment.yml
~~~

## Datasets

### Mozilla Clean Speech clips + Urbonsound 8K noise clips

Mozilla voice dataset: https://commonvoice.mozilla.org/en/datasets

UrbonSound8K dataset: https://urbansounddataset.weebly.com/urbansound8k.html

### Datasets provided by DNS-challenge

Datasets provided by DNS-challenge: https://github.com/microsoft/DNS-Challenge/tree/icassp21/addrir/datasets

To download this dataset, follow the steps outline under https://github.com/microsoft/DNS-Challenge/tree/icassp21/addrir

For this dataset, can utilize the noisyspeech_synthesizer.py to produce (clean,noise,noisy) clips in .wav files. Note that you should change the noisyspeech_synthesizer.cfg to make sure the paths specified and the preprocessing parameters are set.

## Process Dataset

Organize all the clean clips into one folder, and all the noise clips to on folder. Put the clean speech and noise directory paths in process_raw_dataset.py

Need to put the correct path for saving the preprocessed data for training

~~~
python process_raw_dataset.py
~~~

## Train the Model

Modify paths to the locations where the training logs and models to be saved

~~~
python train_model.py
~~~

## Evaluate trained pytorch model on test set

Generate specified number of noisy audio files or generate noisy audio files for all the clean speech files in test set. Then test the denoising effect on the generated noisy signals.

Put the path to store the pytorch model, generated noisy audios, denoised audios and evaluation metrics in test.py

~~~
python test.py
~~~

The objective metrics currently supported is SNR, SegSNR, CSIG, CBAK, COVL, PESQ, STOI; The values for each test clip will be stored in a csv file.
