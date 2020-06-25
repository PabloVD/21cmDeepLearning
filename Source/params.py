#----------------------------------
# Parameters file
# Author: Pablo Villanueva Domingo
# Last update 25/6/20
#----------------------------------

import os


#--- GLOBAL PARAMETERS ---#

# Number of simulations employed
n_sims = 1000

# Number of epochs for training
n_epochs = 20

# Redshifts taken into account. Use only one for now
#redshifts = ["010.16"]
redshifts = ["015.78"]
#redshifts = ["020.18"]

# Number of channels, given by number of redshifts
n_channels = len(redshifts)

# Length of the side of the 3D cube
DIM = 200

# Fraction of the dataset for validation
valid_size = 0.15

# Fraction of the dataset for test
test_size = 0.15

# Size of the batches
batch_size = 30

# Learning rate
learning_rate = 1.e-3

# Weight decay for L2 regularization
weight_decay = 1.e-4


#--- PATHS ---#

# General path
path = os.getcwd()+"/"

# Path to simulations folder
path_simulations = "/projects/QUIJOTE/HI2DMsimulations/"

# Path to fields folder
path_fields = "/tigress/pdomingo/NewFields/"

# Path to outputs folder
path_outputs = "/tigress/pdomingo/Outputs/"

# Sufix for file names indicating some parameters
sufix = "_sims_"+str(n_sims)+"_epochs_"+str(n_epochs)+"_z_"+redshifts[0]+"_batchsize_"+str(batch_size)+"_learningrate_{:.1e}".format(learning_rate)

# Name of the best model
bestmodel = path+"Models/bestmodel"+sufix+".pt"

# Name of best model for the astro net
bestmodel_astro = path+"Models/bestmodel_astro"+sufix+".pt"


#--- OPTIONS ---#

# 1 if data augmentation is used
data_aug = 1

# 1 for training, otherwise it only tests
training = 0

# 1 for loading the previous best model for training
load_model = 0
