#----------------------------------
# Parameter file
# Author: Pablo Villanueva Domingo
# Started 23/9/19
#----------------------------------

import os

#--- GLOBAL PARAMETERS ---#

# Number of simulations employed
n_sims = 1000

# Number of epochs for training
n_epochs = 2

# Redshifts taken into account
#redshifts = ["010.16"]#,"020.18"]
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

#--- PATHS ---#

# General path
path = os.getcwd()+"/" #"/Users/omena/Downloads/21_DeepLearning/"

# Path to Globus files
path_globus = "/tigress/pdomingo/GlobusFiles/"

# Path to fields folder
path_fields = "/tigress/pdomingo/Fields/"   #"/home/pdomingo/DeepLearning21/FieldsNew/"

# Sufix for file names indicating some parameters
sufix = "_sims_"+str(n_sims)+"_epochs_"+str(n_epochs)+"_redshifts_"+str(n_channels)

# Name of best model
bestmodel = path+"bestmodel"+sufix+".pt"

#--- OPTIONS ---#

# 1 if data augmentation is used
data_aug = 0

# 1 for plotting slices (only using UNet)
plot_sli = 1

# 1 for plotting power spectrum (only using UNet)
plot_pow = 1

# 1 for plotting pdf (only using UNet)
plot_pdf = 1

# 1 for training, otherwise it only tests
training = 1

# 1 for loading the previous best model for training
load_model = 0
