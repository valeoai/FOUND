# FOUND
Pytorch implementation of the unsupervised object localization method **FOUND**. More details can be found in the paper:

<div align='center'>

**Unsupervised Object Localization:
Observing the Background to Discover Objects**\
by *Oriane Siméoni, Chloé Sekkat, Gilles Puy, Antonin Vobecky, Eloi Zablocki and Patrick Pérez*

[![Arxiv](http://img.shields.io/badge/paper-arxiv.2212.07834-B31B1B.svg)](https://arxiv.org/abs/2212.07834)

<div>
  <img width="100%" alt="FOUND visualizations" src="data/examples/found_examples.png">
</div>

</div>

\
If you use our **FOUND** code or framework in your research, please consider citing:


```
@article{simeoni2022unsupervised,
  author    = {Siméoni, Oriane and Sekkat, Chloé and Puy, Gilles and Vobecky, Antonin and Zablocki, Éloi and Pérez, Patrick},
  title     = {Unsupervised Object Localization: Observing the Background to Discover Objects},
  journal   = {CoRR},
  volume    = {abs/2212.07834},
  year      = {2022},
  ee        = {http://arxiv.org/abs/2212.07834}
}
```

## Overview

- [Presentation](#found)
- [Installation](#installation-of-found)
- [Using FOUND](#usage-of-found)
- [Evaluation: Saliency object detection](#saliency-object-detection)
- [Evaluation: Unsupervised object discovery](#unsupervised-object-discovery)

## Installation of FOUND

### Environment installation

This code was implemented and tested with python 3.7, PyTorch 1.8.1 and CUDA 11.1. Please install [PyTorch](https://pytorch.org/). In order to install the additionnal dependencies, please launch the following command:

```bash
# Create conda environment
conda create -n found python=3.7
conda activate found

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

