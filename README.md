# Hackathon
Let's red team [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) and [Whisper](https://github.com/openai/whisper)!

## Instructions
First, clone this repo locally (or on the cluster.)
```bash
git clone https://github.com/angelo-ortiz/hackathon.git
```
Then, install the requirements for MMS
```bash
cd hackathon
conda env create -f environment.yml
```

Then, install the MMS model
```bash
wget -P ./models_new 'https://dl.fbaipublicfiles.com/mms/asr/mms1b_l1107.pt'
```

Then get a sample audio file (or record one yourself, call it audio.wav (16khz, please) and put it in the directory ./audio_samples)

```bash
mkdir ./audio_samples
wget -P ./audio_samples/ 'https://datasets-server.huggingface.co/assets/google/fleurs/--/en_us/train/0/audio/audio.mp3'
sox audio_samples/audio.mp3 audio_samples/audio.wav
```

Then run the model on the audio sample
```bash
python ./asr/infer/mms_infer.py --model "./mms1b_fl102.pt" --lang "eng" --audio "./audio_samples/audio.wav"
```


## Links
- [MMS 1B model](https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt)
- [ASR example](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/asr/tutorial/MMS_ASR_Inference_Colab.ipynb)
