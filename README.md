# 21cmDeepLearning
Python codes to extract the density field from a 21 cm intensity field, making use of a convolutional neural network (CNN) with the UNet architecture. The simulations, performed with [21cmFAST](https://github.com/andreimesinger/21cmFAST/commits/master), can be obtained using [Globus](https://www.globus.org/data-transfer).
The files included are the following:

- GlobusTransfer.py: code to get the simulations from the remote repository where they are hosted through Globus.

- Dataloader.py: convert the binary files to a proper format and take the slices.

- DM2HI.py: train and test the network.

- Source/functions.py: includes some useful functions, such as routines for loading the data and training the net.

- Source/plot_routines.py: includes some plotting routines.

- Source/params.py: parameters to be set by the user.

- Source/Unet.py: CNN UNet architecture.