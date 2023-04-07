# FOUND
Pytorch implementation of the unsupervised object localization method **FOUND**. More details can be found in the paper:

<div align='center'>

<h3>Unsupervised Object Localization: Observing the Background to Discover Objects</h3>
by <i>Oriane Siméoni</i>, <i>Chloé Sekkat</i>, <i>Gilles Puy</i>, <i>Antonin Vobecky</i>, <i>Eloi Zablocki</i> and <i>Patrick Pérez</i> <br>
<b> CVPR 2023 </b>
<h4 align="center">
  | <a href="https://valeoai.github.io/blog/publications/found/">project page</a> |
  <a href="https://arxiv.org/abs/2212.07834">arXiv</a> |
  <a href="https://huggingface.co/spaces/osimeoni/FOUND">gradio</a> |
</h4>

<div>
  <img width="80%" alt="FOUND visualizations" src="data/examples/found_examples.png">
</div>

</div>


\
If you use our **FOUND** code or framework in your research, please consider citing:


```
@inproceedings{simeoni2023found,
  author    = {Siméoni, Oriane and Sekkat, Chloé and Puy, Gilles and Vobecky, Antonin and Zablocki, Éloi and Pérez, Patrick},
  title     = {Unsupervised Object Localization: Observing the Background to Discover Objects},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}},
  year      = {2023},
}
```

## Updates

- [Apr. 2023] Release of training code [available here](#training-of-found)
- [Mar. 2023] Gradio space is available ([link](https://huggingface.co/spaces/osimeoni/FOUND))
- [Feb. 2023] FOUND is accepted to CVPR23 !
- [Dec. 2022] First release

## Overview

- [Presentation](#found)
- [Installation](#installation-of-found)
- [Using FOUND](#usage-of-found)
- [Evaluation: Saliency object detection](#saliency-object-detection)
- [Evaluation: Unsupervised object discovery](#unsupervised-object-discovery)
- [Training of FOUND](#training-of-found)
- [Acknowledgments](#acknowledgments)

## Installation of FOUND

### Environment installation

This code was implemented and tested with python 3.7, PyTorch 1.8.1 and CUDA 11.1. Please install [PyTorch](https://pytorch.org/). In order to install the additionnal dependencies, please launch the following command:

```bash
# Create conda environment
conda create -n found python=3.7
conda activate found

# Example of pytorch installation
pip install torch===1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
```

Please install also DINO [paper](https://arxiv.org/pdf/2104.14294.pdf). The framework can be installed using the following commands:
```bash
git clone https://github.com/facebookresearch/dino.git
cd dino; 
touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```

## Usage of FOUND 

We provide here the different command lines in order to repeat all results provided in our paper. 

### Application to one image

Using the following command it is possible to apply and visualize our method on one single image.

```bash
python main_visualize.py --img-path /datasets_local/VOC2007/JPEGImages/000030.jpg
```

### Saliency object detection

We evaluate our method *FOUND* for the saliency detection on the datasets 
- [DUT-OMRON](http://saliencydetection.net/dut-omron/): `--dataset-eval DUT-OMRON`
- [DUTS-TEST](http://saliencydetection.net/duts/): `--dataset-eval DUTS-TEST`
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html): `--dataset-eval ECSSD`.

Please download those datasets and provide the dataset directory using the argument `--dataset-dir`. 
The parameter `--evaluation-mode` allow you to choose between `single` and `multi` setup and `--apply-bilateral` can be added to apply the bilateral solver. Please find here examples on the dataset `ECSSD`.

```bash
export DATASET_DIR=data/ # Put here your directory

# ECSSD, single mode, with bilateral solver post-processing
python main_found_evaluate.py --eval-type saliency --dataset-eval ECSSD --evaluation-mode single --apply-bilateral --dataset-dir $DATASET_DIR
# same but multi mode
python main_found_evaluate.py --eval-type saliency --dataset-eval ECSSD --evaluation-mode multi --apply-bilateral --dataset-dir $DATASET_DIR
```

### Unsupervised object discovery

In order to evaluate on the unsupervised object discovery task, we follow the framework used in our previous work [LOST](https://github.com/valeoai/LOST).
The task is implemented for the following datasets, please download the benckmarks and put them in the folder `data/`.
- [VOC07](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC07`
- [VOC12](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC12`
- [COCO20k](https://cocodataset.org/#home): `--dataset-eval COCO20k`

```bash
export DATASET_DIR=data/ # Put here your directory

# VOC07
python main_found_evaluate.py --eval-type uod --dataset-eval VOC07 --evaluation-mode single --dataset-dir $DATASET_DIR
# VOC12
python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR
# COCO20k
python main_found_evaluate.py --eval-type uod --dataset-eval COCO20k --evaluation-mode single --dataset-dir $DATASET_DIR
```

## Training of FOUND

In order to train a FOUND model, please start by [installing](#installation-of-found) the framework. If already installed, please run again 

```bash
# Create conda environment
conda activate found

# Install dependencies
pip install -r requirements.txt
```

The training is performed on the dataset [DUTS-TR](http://saliencydetection.net/duts/) that should be put in the directory `data`. 

Then the training can be launched using the following command. Visualizations and training curves can be observed using tensorboard.
```bash
export DATASET_DIR=data/ # Root directory of all datasets, both training and evaluation

python main_found_train.py --dataset-dir $DATASET_DIR
```

Once the training done, you can launch the evaluation using the scripts `evaluate_saliency.sh` and `evaluate_uod.sh` with the commands:

```bash
export MODEL="outputs/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt"

# Evaluation of saliency detection
source evaluate_saliency.sh $MODEL $DATASET_DIR single
source evaluate_saliency.sh $MODEL $DATASET_DIR multi

# Evaluation of unsupervised object discovery
source evaluate_uod.sh $MODEL $DATASET_DIR
```

## Acknowledgments

This repository was build on the great works [SelfMask](https://github.com/NoelShin/selfmask), [TokenCut](https://github.com/YangtaoWANG95/TokenCut) and our previous work [LOST](https://github.com/valeoai/LOST). Please, consider acknowledging these projects.
