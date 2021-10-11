# DeepLearning
Materials for my Deep Learning course 


## Setting up the python environment

As in most of the machine learning applications, you will be working with python using jupyter notebooks or jupyter lab. So first you have to set up a proper python environment. I strongly encourage you to use some form of a virtual environment. I recommend the Anaconda or its smaller subset miniconda. Personally I recommend using 
[mambaforge](https://github.com/conda-forge/miniforge#mambaforge). 
After installing `mambaforge` create a new virtual environment deeplearning (or any other name you want):

```
conda create -n deeplearning python=3.9
```
Then activate the environment  by running
```
conda activate deeplearning
```
Now you can install required packages (if you are using Anaconda some maybe already installed):

```
mamba install  jupyterlab jupytext myst-nb
mamba install numpy scipy  matplotlib
```
If didn't install `mamba` then you can subsitute `conda` for `mamba`. I tend to use `mamba` as it is markedly faster then `conda`.  

You will also need PyTorch. The installation instruction are [here](https://pytorch.org/get-started/locally/). If you have a relatively new NVidia GPU you should install a GPU version.

## MyST format

The notebooks in the repository are stored in [MyST (Markedly Structured Text Format)](https://myst-parser.readthedocs.io/en/latest/) format. Thanks to the `jupytext` package you can open them right in the jupyter lab, by clicking the file name with righthand mouse button and choosing `open with` and then `Notebook`. If you are using jupyter notebook the you have to convert them prior to opening by running   
```shell
jupytext --to notebook <md file name>
```