# 21cmDeepLearning
Python codes to extract the density field from a 21 cm intensity field. The simulations can be obtained using [Globus](https://www.globus.org/data-transfer).


- GlobusTransfer.py: code to get the simulations from the remote repository where they are hosted through Globus.

- Dataloader.py: convert the binary files to a proper format and take the slices.

- DM2HI.py: train and test the network.