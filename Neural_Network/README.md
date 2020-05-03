# Neural network - an intercepted message originating from Mustafar
#### Dennis Bakhuis - May the 4th be with you 2020
https://linkedin.com/in/dennisbakhuis/

A annotated Jupyter Notebook in which I create a neural network from scratch, using only Numpy. After a short explanation, two classes are created which represent our Artificial Neural Network (ANN). 

After creation of the classes, I give the implementation a spin and try it out on:
- linear regression (same as the logistric regression tutorial)
- logistic regression (Titanic sank yet again)
- complex 2d functions
- cat vs not-cat classifier (From Andrew Ng's Coursera class)
- **an intercepted transmission from Mustafar**

## Prerequisits for this tutorial:
- a working anaconda / miniconda environment.

## Create the a new Python environment and download the required packages:
1) open a shell
2) conda create --name tutor python=3.7
3) conda activate tutor
4) pip install numpy pandas jupyterlab matplotlib h5py scikit-image tqdm

For completeness, I also provided a requirements.txt, but the above should work. The packages h5py and scikit-image are only needed when you want to recreate the data.\
If you want tqdm (a nifty progressbar) to work in jupyter lab, please check the website of tqdm, as you need two more commands and maybe nodejs.

## Open Jupyter Lab in the working directory which has the Notebook and Data:
1) jupyter lab
2) In jupyter lab, open the notebook named: Neural_Network.ipynb

