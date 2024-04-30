# Hackathon
Let's red team [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) and [Whisper](https://github.com/openai/whisper)!

## Instructions
First, clone this repo locally (or on the cluster.)
```bash
git clone https://github.com/angelo-ortiz/hackathon.git
```
Then, install the requirements
```bash
cd hackathon
conda env create -f environment.yml
```

Then, install the model
```bash
wget -P ./models_new 'https://dl.fbaipublicfiles.com/mms/asr/mms1b_l1107.pt'
```

The get a sample audio file (or record one yourself, and put it in the directory ./audio_samples)

```bash
wget -P ./audio_samples/ 'https://datasets-server.huggingface.co/assets/google/fleurs/--/en_us/train/0/audio/audio.mp3'
ffmpeg -y -i ./audio_samples/audio.mp3 -ar 16000 ./audio_samples/audio.wav
```

## Links
- [MMS 1B model](https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt)
- [ASR example](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/asr/tutorial/MMS_ASR_Inference_Colab.ipynb)
