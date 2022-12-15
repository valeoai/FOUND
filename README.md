# FOUND 
Pytorch implementation of the unsupervised object localization method **FOUND**. More details can be found in the paper:

**Unsupervised Object Localization:
Observing the Background to Discover Objects**, arxiv 2022 [[arXiv](TBC)]  
by *Oriane Siméoni, Chloé Sekkat, Gilles Puy, Antonin Vobecky, Eloi Zablocki and Patrick Pérez*

<div>
  <img width="100%" alt="FOUND visualizations" src="data/examples/found_examples.png">
</div>  

\
If you use our **FOUND** code or framework in your research, please consider citing:


```
@inproceedings{simeoni2022unsupervised,
   title = {Unsupervised Object Localization: Observing the Background to Discover Objects},
   author = {Oriane Sim\'eoni and  Chlo\'e Sekkat and Gilles Puy and Antonin Vobecky and Eloi Zablocki and Patrick P\'erez},
   journal = {},
   month = {Decembre},
   year = {2022}
}
```

## Installation of FOUND
### Environment installation

This code was implemented with python 3.7, PyTorch 1.8.1 and CUDA 11.1. Please install [PyTorch](https://pytorch.org/). In order to install the additionnal dependencies, please launch the following command:

```bash
# Create conda environment
conda create -n found python=3.7

# Install dependencies
conda activate found
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

Using the following command it is possible to apply our method to one image

```bash
python main_visualize.py --img-path /datasets_local/VOC2007/JPEGImages/000030.jpg
```

### Saliency object detection

We evaluate our method *FOUND* for the saliency detection on the datasets `DUT-OMRON`, `DUTS-TEST`, `ECSSD`. 
Please download those dataset from `http://saliencydetection.net/dut-omron/`, `http://saliencydetection.net/duts/` and `https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html` respectively. 
The parameter `--evaluation-mode` allow to choose between `single` and `multi` setup and `--apply-bilateral` can be added to apply the bilateral solver. Please find here examples on the dataset `ECSSD`.

```bash
python main_found_evaluate.py --eval-type saliency --dataset-eval ECSSD --evaluation-mode single --apply-bilateral
python main_found_evaluate.py --eval-type saliency --dataset-eval ECSSD --evaluation-mode multi --apply-bilateral
```