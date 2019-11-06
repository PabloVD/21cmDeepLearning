#----------------------------------
# Parameter file
# Author: Pablo Villanueva Domingo
# Started 23/9/19
#----------------------------------

import os

#--- GLOBAL PARAMETERS ---#

# Number of simulations employed
n_sims = 50

# Number of epochs for training
n_epochs = 10

# Redshifts taken into account
#redshifts = ["010.16","015.78","020.18"]
redshifts = ["015.78"]

# Number of channels, given by number of redshifts
n_channels = len(redshifts)

# Length of the side of the 3D cube
DIM = 200

# Fraction of the dataset for validation
valid_size = 0.15

# Fraction of the dataset for test
test_size = 0.15

# Size of the batches
batch_size = 15

# Path
path = os.getcwd()+"/" #"/Users/omena/Downloads/21_DeepLearning/"

# Sufix for file names indicating some parameters
sufix = "_n_sims_"+str(n_sims)+"n_epochs_"+str(n_epochs)+"n_redshifts"+str(n_channels)

#--- OPTIONS ---#

# 1 if data augmentation is used
data_aug = 1

# 1 for plotting slices (only using UNet)
plotsli = 1

# 1 for training, otherwise it only tests
training = 1

# 1 for loading the previous best model for training
load_model = 0
