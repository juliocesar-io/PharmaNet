# **PharmaNet: Pharmaceutical discovery with deep recurrent neural networks.**

This repository provides a PyTorch implementation of PharmaNet, presented in the paper [PharmaNet: Pharmaceutical discovery with deep recurrent neural networks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241728). PharmaNet is a novel approach for the task of Drug-Protein interaction based on Recurrent Neural Networks to analyze SMILES representations drugs.

# **Overview**
<p align="center"><img src="Overview.png" /></p>

## Download the pretrained models

Get the pre trained models from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/juliocesar-io/PharmaNet
```

Then, create a new conda environment:  

```bash
mamba create -n PharmaNet python=3.8
```

## CPU only installation

Activate the environment:

```bash
mamba activate PharmaNet
```

Install the dependencies:

```bash
mamba install pytorch==1.9.0 torchvision==0.10.0 -c pytorch
pip install pandas tqdm matplotlib scikit-image scikit-learn
```

## GPU installation

If you have a CUDA enabled GPU, you can install the dependencies for GPU:

Activate the environment:

```bash
mamba activate PharmaNet
```

Install the dependencies:

```bash
mamba install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install pandas tqdm matplotlib scikit-image scikit-learn
```