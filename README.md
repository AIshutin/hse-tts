# ASR project barebones

## Installation guide

```
pip install -r ./requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
wget https://www.openslr.org/resources/11/3-gram.arpa.gz -O 4-gram.arpa.gz
gzip -d 4-gram.arpa.gz
wget https://openslr.elda.org/resources/11/librispeech-lexicon.txt -O librispeech-lexicon.txt
wget https://www.openslr.org/resources/11/librispeech-vocab.txt -O librispeech-vocab.txt
```

## Usage guide

To train:
```shell
python3 train.py -c hw_asr/configs/all_deepspeech_small.json
```

To measure quality:
```shell
gdown "https://drive.google.com/file/d/1DftKTAVPW7tHauL8H9cTzph3ozMMkE3F/view?usp=sharing" -O default_test_model/checkpoint.pth --fuzzy
python3 test.py -c default_test_model/config.json -r default_test_model/checkpoint.pth -b 1
```

To run unit tests:
```shell
python3 -m unittest hw_asr
```


## What's done?

- Augmentations (Pitch Shift, Gaussian Noise, Gaussian Noise SNR, Time Stretch)
- Deepspeech-like and Deepspeech2-like architectures
- Sortagrad (not used in the final model)
- Length-based batch-loader (not used in the final model)
- BPE encoding (not used in the final model)
- Audio logging
- LRFinder config
- Beam search (BS) and beam search + language model (LM)

See [report](https://wandb.ai/aishutin/asr_project/reports/---Vmlldzo1Nzg4MzQz?accessToken=bjh11ugotng1gel7btqq822g7rrpxjzqdah0azzytfs6z3qxau13osht7cr589is) (in Russian) for more information

## Results

**Librispeech: test-clean**

|metric|argmax|BS    |BS+LM|
|------|------|------|-----|
|CER   |45.46%|      |     |
|WER   |94.96%|      |     |


# HW information

## Recommended implementation order

You might be a little intimidated by the number of folders and classes. Try to follow this steps to gradually undestand
the workflow.

1) Test `hw_asr/tests/test_dataset.py`  and `hw_asr/tests/test_config.py` and make sure everythin works for you
2) Implement missing functions to fix tests in  `hw_asr\tests\test_text_encoder.py`
3) Implement missing functions to fix tests in  `hw_asr\tests\test_dataloader.py`
4) Implement functions in `hw_asr\metric\utils.py`
5) Implement missing function to run `train.py` with a baseline model
6) Write your own model and try to overfit it on a single batch
7) Implement ctc beam search and add metrics to calculate WER and CER over hypothesis obtained from beam search.
8) ~~Pain and suffering~~ Implement your own models and train them. You've mastered this template when you can tune your
   experimental setup just by tuning `configs.json` file and running `train.py`
9) Don't forget to write a report about your work
10) Get hired by Google the next day

## Before submitting

0) Make sure your projects run on a new machine after complemeting the installation guide or by 
   running it in docker container.
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checkpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize

## TODO

These barebones can use more tests. We highly encourage students to create pull requests to add more tests / new
functionality. Current demands:

* Tests for beam search
* README section to describe folders
* Notebook to show how to work with `ConfigParser` and `config_parser.init_obj(...)`
