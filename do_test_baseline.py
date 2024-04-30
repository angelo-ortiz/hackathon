# Hackaton CoML - T.Hueber - 2024
#########

#############
# HOW TO INSTALL
# Create a venv (python -m venv /path_to_venv/
# Activate venv (source /path_to_venv/bin/activate.sh)
# git clone https://github.com/UDASE-CHiME2023/baseline.git
# export PYTHONPATH={the path that you stored the github repo}:$PYTHONPATH
# cd baseline
# python -m pip install --user -r requirements.txt (I think I installed only the asteroid==0.6.0)
# pip install transformers (for HuggingFace) 
# Prepare short audio clips (16bits, WAV, 16KhZ, 1 channel) --> add the path to input_audio_filename
# Run this script 
# Output is stored in basename_denoised.wav (basename is extracted from input_audio_filename)
#################################

import torch
import torchaudio
import baseline.utils.mixture_consistency as mixture_consistency
import baseline.models.improved_sudormrf as improved_sudormrf
import soundfile as sf
import pdb
import numpy as np
import os
from transformers import Wav2Vec2ForCTC, AutoProcessor

# Config
########
input_audio_filename = './foo4.wav'
model_id = "facebook/mms-1b-all"
target_lang = "eng"

# Run
#####
print("Denoising %s ..." % input_audio_filename)
model = improved_sudormrf.SuDORMRF(
        out_channels=256,
        in_channels=512,
        num_blocks=8,
        upsampling_depth=7,
        enc_kernel_size=81,
        enc_num_basis=512,
        num_sources=2,
    )

# Speech enhancement (baseline of CHIME7)
model.load_state_dict(torch.load('./pretrained_checkpoints/remixit_chime_adapted_student_using_vad.pt'))
model = torch.nn.DataParallel(model)

input_mix, _ = torchaudio.load(input_audio_filename) 
input_mix = input_mix.unsqueeze(1)#.cuda() 

input_mix_std = input_mix.std(-1, keepdim=True)
input_mix_mean = input_mix.mean(-1, keepdim=True)
input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

in_samples = input_mix.detach().numpy() # useful for ASR with MMS

estimates = model(input_mix)
estimates = mixture_consistency.apply(estimates, input_mix)

estimates = estimates.detach().numpy()

out_samples = estimates[0,0,:]/np.max(input_mix.detach().numpy())
sf.write(os.path.basename(input_audio_filename) + '_denoised.wav', out_samples, 16000, subtype='PCM_16')
print("done")
############################

print("ASR ...")
processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True)

inputs = processor(in_samples[0,0,:], sampling_rate=16_000, return_tensors="pt")
inputs_denoised = processor(out_samples, sampling_rate=16_000, return_tensors="pt")
outputs = model(**inputs).logits
outputs_denoised = model(**inputs_denoised).logits

ids = torch.argmax(outputs, dim=-1)[0]
ids_denoised = torch.argmax(outputs_denoised, dim=-1)[0]
transcription= processor.decode(ids)
transcription_denoised = processor.decode(ids_denoised)
print("Transcription (original signal): " + transcription)
print("Transcription (denoised signal): " + transcription_denoised)

##END
########