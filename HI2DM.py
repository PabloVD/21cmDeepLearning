#-------------------------------------------
# CNN to predict 21cm field from a DM field
# Author: Pablo Villanueva Domingo
# Started: 23/9/19
# Last modification: 15/10/19
#-------------------------------------------

import time, datetime
from Source.functions import *
from Source.Unet import UNet2

#--- MAIN ---#

time_ini = time.time()
if not os.path.exists(path+"Plots"):
    os.system("mkdir "+path+"Plots")

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
#model = UNet(n_channels=n_channels, n_classes=n_channels)
model = UNet2(n_channels, n_channels)
lossfunc = nn.MSELoss()

if train_on_gpu:
    model.cuda()

network_total_params = sum(p.numel() for p in model.parameters())
print('Total number of parameters in the model = %d'%network_total_params)
print("Data loaded. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))

# Print the memory (in Gb) being used now:
import psutil
process = psutil.Process()
print("Memory being used (Gb):",process.memory_info().rss/1.e9)

# Train the net
if training:
    print("Learning...")
    # Load the best model so far if wanted
    if load_model:
        if os.path.exists(bestmodel):
            print("Loading previous best model")
            state_dict = torch.load(bestmodel)
            model.load_state_dict(state_dict)
        else:
            print("No previous model to load")
    train_losses, valid_losses = learning_loop(model,train_loader,valid_loader,lossfunc,n_epochs)

# Test the net
print("Testing...")
true_targets, predicted_targets, test_loss = testing_loop(model,test_loader,lossfunc)

#--- PLOTS ---#

if training:
    # Show validation/training trend
    loss_trend(train_losses,valid_losses,test_loss)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
