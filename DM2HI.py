#-------------------------------------------
# CNN to predict 21cm field from a DM field
# Author: Pablo Villanueva Domingo
# Started: 23/9/19
# Last modification: 15/10/19
#-------------------------------------------

import time
from Source import UNet
from Source.functions import *

#--- MAIN ---#

time_ini = time.time()

# Load fields and convert to tensors
print("Loading dataset...")
inputs = normalize_field(load_field("dTb"))
targets = normalize_field(load_field("delta"))
tensor_x, tensor_y = torch.from_numpy(inputs), torch.from_numpy(targets)
print("Shape data: ",tensor_x.shape,tensor_y.shape)
totaldata = utils.TensorDataset(tensor_x.float(),tensor_y.float())

# Split training and validation sets
train_loader, valid_loader, test_loader = split_datasets(totaldata)

# Choose model and loss function
model = UNet(n_channels=n_channels, n_classes=n_channels)
lossfunc = nn.MSELoss()

# Train the net
if training:
    print("Learning...")
    # Load the best model so far if wanted
    if load_model:
        state_dict = torch.load('bestmodel.pt')
        model.load_state_dict(state_dict)
    train_losses, valid_losses = learning_loop(model,train_loader,valid_loader,lossfunc,n_epochs)

# Test the net
print("Testing...")
true_targets, predicted_targets, test_loss = testing_loop(model,test_loader,lossfunc)

#--- PLOTS ---#

if training:
    # Show validation/training trend
    loss_trend(train_losses,valid_losses,test_loss)


print("Minutes elapsed:",(time.time()-time_ini)/60.)
