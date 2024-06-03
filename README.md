# HUnEE-B

HUnEE-B: bee**H**ive monitoring **U**niversal p**E**rformanc**E** **B**enchmark

# Intro

> tl;dr: HUnEE-B is a benchmark suite for developing general-purpose audio representations that generalize across various tasks in automatic acoustic beehive monitoring. HUnEE-B's first version evaluates eleven models across four critical tasks: beehive state detection, beehive strength assessment, buzzing identification, and beekeeper voice activity detection.

![Alt text](./assets/huneeb_schematics.png?raw=true "Benchmark overview")

# Citing

```bibtex
@todo
```

# Call for contributions

We are looking for contributions to extend the number of available downstream tasks. If you have a dataset and a task that you would like to see implemented in HUnEE-B, please open an issue or a pull request.

# How to use it

## Installation

1. Clone this repository and Install the package

```bash
$ git clone https://github.com/Hguimaraes/beeSSL
$ cd HUnEE-B
$ pip install -e .
```

2. Download the prepared datasets using this [[LINK](https://www.dropbox.com/scl/fo/al2ht1qslo7xodyetrior/ANkND8v_1xIw_u9RwM4FSU0?rlkey=hy82eb25z6hoyu57pwnhj1ygi&st=yez53216&dl=0)]. Extract in a folder of your choice.

3. (Optional) Download the pre-trained baseline models [[HERE](https://www.dropbox.com/scl/fo/j82cjackgplhxpv84s5dd/ANYFmFxhF2cNZHfItRSsIvg?rlkey=9kf6kfv11qglvrv2kmml3a9bs&st=p69lms5r&dl=0)]. If you do so, create a folder `result/pretrain/` and extract the files there.

## Reproducing the results from the paper

An example of how to train and evaluate the baseline models for the beehive strength task is shown below:

```bash
for seed in 1234 5678 9012 3456 7890 6015 2017 1984 1776 2024;
do
    python run_downstream.py -m train -d beehive_strength -u fbank -n bstr_fbank/$seed -o"data_root=/PATH/TO/DATASET" --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u fbank -n bstr_fbank/$seed -o"data_root=/PATH/TO/DATASET" --seed $seed
done
```

where,<br>
`-m`: flag for the mode (train or evaluate) <br>
`-d`: flag for the downstream task <br>
`-u`: flag for the upstream model <br>
`-n`: flag for the experiment name <br>
`-o`: flag for the experiment options <br>
`--seed`: flag for the random seed <br>

## Extending to your own Upstream models

For an easy setup, take a look on the `upstream/cav_mae/` folder. There are three main steps to extend HUnEE-B to your own upstream models:

1. Create a new folder with the name of your in `upstream/` and create three files: `__init__.py`, `expert.py` and `hubconf.py`.

2. Implement your model in expert.py. The UpstreamExpert class, which you need to implement, is a subclass of torch.nn.Module. In the forward process, you will receive a batch of audio samples. Your method should return a dictionary containing a list of embeddings for the samples per layer, as shown in the following example:

```python
hidden_states = self.model(wavs) # L x [B x F x T]
return {"hidden_states": hidden_states}
```

If your method returns only the embeddings from the last layer, return a list with only one element.


3. Implement the `hubconf.py` file to handle the UpstreamExpert instantiation. If your model depends on a config file, we recommend you to use the hyperyaml from speechbrain.

Lastly, import your model in the `hub.py` file and you are ready to go! To use your model, you can use the `-u` flag in the `run_downstream.py` script with the name of the function that you created in the hubconf.py file.