#-------------------------------------------------------------
# CNN to predict the matter density field from a 21 cm field
# Main script for training and testing the network
# Author: Pablo Villanueva Domingo
# Last update: 25/6/20
#-------------------------------------------------------------

import time, datetime, psutil
from Source.functions import *
from Source.nets import UNet
from Source.plot_routines import loss_trend
#from torchsummary import summary

#--- MAIN ---#

time_ini = time.time()

# Make some directories if they don't exist yet
if not os.path.exists(path+"Plots"):
    os.mkdir(path+"Plots")
if not os.path.exists(path+"Models"):
    os.mkdir(path+"Models")
if not os.path.exists(path_outputs):
    os.mkdir(path_outputs)
if not os.path.exists(path_outputs+"Outputs"+sufix):
    os.mkdir(path_outputs+"Outputs"+sufix)

# Load fields, normalize and convert to tensors
print("Loading dataset...")
inputs = load_field("dTb")
targets = load_field("delta")
inputs = normalize_field(inputs)
targets = normalize_field(targets)

tensor_x, tensor_y = torch.from_numpy(inputs), torch.from_numpy(targets)
print("Shape data: ",tensor_x.shape,tensor_y.shape)
totaldata = utils.TensorDataset(tensor_x.float(),tensor_y.float())

# Split training and validation sets
if training:
    train_loader, valid_loader, test_loader = split_datasets(totaldata)
else:
    test_loader = utils.DataLoader(totaldata, batch_size=batch_size)

# Choose model and loss function
model = UNet(n_channels, n_channels)
#summary(model,(1,DIM,DIM))
lossfunc = nn.MSELoss()

if train_on_gpu:
    model.cuda()

network_total_params = sum(p.numel() for p in model.parameters())
print('Total number of parameters in the model = %d'%network_total_params)
print("Data loaded. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))

# Print the memory (in GB) being used now:
process = psutil.Process()
print("Memory being used (GB):",process.memory_info().rss/1.e9)

# Train the net
if training:
    print("Learning...")
    # Load the best model so far if wanted
    if load_model:
        if os.path.exists(bestmodel):
            print("Loading previous best model")
            state_dict = torch.load(bestmodell, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print("No previous model to load")

    train_losses, valid_losses = learning_loop(model,train_loader,valid_loader,lossfunc,n_epochs)

    # Plot the validation/training trend
    loss_trend(train_losses,valid_losses)


# Test the net
print("Testing...")
true_targets, predicted_targets, test_loss = testing_loop(model,test_loader,lossfunc)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
