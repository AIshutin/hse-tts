# TTS Project

Based on this [template](https://github.com/WrathOfGrapes/asr_project_template) and [this FastSpeech 1 implementation](https://github.com/xcmyz/FastSpeech).

## Installation guide

```
pip install -r ./requirements.txt
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
mkdir data ; mkdir data/datasets ; mkdir data/datasets/ljspeech
mv alignments data/datasets/ljspeech/alignments

gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)

mv mels data/datasets/ljspeech

python3 preprocess.py -c tts/configs/preprocess.json

gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt


```

## Usage guide

To train:
```shell
python3 train.py -c tts/configs/all_deepspeech_small.json
```

To measure quality:
```shell
gdown "https://drive.google.com/file/d/1DftKTAVPW7tHauL8H9cTzph3ozMMkE3F/view?usp=sharing" -O default_test_model/checkpoint.pth --fuzzy
python3 test.py -c default_test_model/config.json -r default_test_model/checkpoint.pth -b 1
```