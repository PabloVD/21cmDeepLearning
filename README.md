# 21cmDeepLearning

Python codes to extract the underlying matter density map from a 21 cm intensity field, making use of a convolutional neural network (CNN) with the U-Net architecture. Implemented in Pytorch. The astrophysical parameters of the simulations can also be predicted with a secondary CNN. The simulations of matter density and 21 cm maps have been performed with the code [21cmFAST](https://github.com/andreimesinger/21cmFAST/commits/master).

See the paper [arXiv:XXXX.XXXX](arXiv:XXXX.XXXX) for more details.

## Description of the scripts

The files included are the following:

* `Dataloader.py`: convert the binary files from the simulations to numpy arrays and store 2D slices.

* `HI2DM.py`: main script for training and testing the U-Net network to recover the matter density field from 21 cm maps.

* `HI2Astro.py`: script for training and testing a secondary CNN to predict the astrophysical parameters of the 21 cm maps. It is optional to employ the pre-trained weights of the encoder in the U-Net, trained running `HI2DM.py`. 

* `Plotter.py`: driver for plotting several outputs and statistics. Most of routines are defined in `Source/plot_routines.py`.

* `Saliency_astro.py`: script to compute the saliency maps of the astrophysical network (see e.g. [arXiv:1312.6034](https://arxiv.org/abs/1312.6034)).

In the folder `Source`, several auxiliary routines are defined:

* `params.py`: parameters to be set by the user, such as number of epochs, number of simulations, learning rate, etc.

* `nets.py`: includes the definition of the networks architectures, the U-Net and the astrophysical network.

* `functions.py`: includes some useful functions, such as routines for loading the data and training the net.

* `plot_routines.py`: includes some plotting routines and function to compute and plot statistics such as the PDF and the power spectrum.

## Requisites

The libraries required for training the CNNs are
* numpy
* pytorch
* matplotlib

For some plots and statistics, the following packages are also needed:
* scipy
* sklearn
* powerbox

## Usage

You may want to run the scripts in the following order:
1. Run the 21cmFAST simulations and store them in `path_simulations` (path defined in params.py).
2. Run `Dataloader.py` to extract the relevant fields for the required redshifts.
3. Run `HI2DM.py` to train the U-Net for predicting the matter density field given a 21 cm map.
4. Run `Plotter.py` for plotting several statistics and samples of the maps.
5. Run `HI2Astro.py` to train the secondary CNN to predict the astrophysical parameters.
6. Run `Saliency_astro.py` to compute the saliency maps of the astrophysical network.

## Contact

For comments, questions etc. you can reach me at <pablo.villanueva@ific.uv.es>
