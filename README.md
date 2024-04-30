# Hackathon
Let's red team [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) and [Whisper](https://github.com/openai/whisper)!

## Instructions
First, clone this repo locally (or on the cluster.)
```bash
cd /path/where/to/save/this/repo  # e.g. in Mac: cd /Users/myself/Documents
git clone https://github.com/angelo-ortiz/hackathon.git
```
Then, install the requirements
```bash
cd hackathon
conda env create -f environment.yml
conda activate hack
```
The final requirement is fairseq. To install it, do the following
```bash
cd /path/where/to/save/fairseq  # e.g. in Mac: cd /Users/myself/Documents
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
```
Then, install the model
```bash
wget -P ./models_new 'https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt'
```

Then get a sample audio file (or record one yourself, call it audio.wav (16khz, please) and put it in the directory ./audio_samples)
```bash
cd 
wget -P ./audio_samples/ 'https://datasets-server.huggingface.co/assets/google/fleurs/--/en_us/train/0/audio/audio.mp3'
ffmpeg -y -i ./audio_samples/audio.mp3 -ar 16000 ./audio_samples/audio.wav
```

Before running the model, modify the following line in `/path/to/fairseq/examples/mms/asr/config/infer_common.yaml`.
More importantly, prepend your home directory (e.g. in Mac: `/Users/{env:USER}`)
```bash
dir: /Users/${env:USER}/checkpoint/${env:USER}/${env:PREFIX}/${common_eval.results_path}
```
If you do not have a GPU, you also need to add the following lines to the same file:
```bash
common:
  cpu: true
```

Finally, you can run the model in the notebook [asr_with_mms.ipynb](asr_with_mms.ipynb)


## Links
- [MMS 1B model](https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt)
- [ASR example](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/asr/tutorial/MMS_ASR_Inference_Colab.ipynb)
