#----------------------------------------------------------------
# CNN to predict the astrophysical parameters from a 21 cm field
# It can employ the encoder of the pre-trained U-Net
# Author: Pablo Villanueva Domingo
# Last update: 25/6/20
#----------------------------------------------------------------

import time, datetime, psutil
from Source.functions import *
from Source.nets import Encoder, AstroNet
from Source.plot_routines import loss_trend, param_plot
#from torchsummary import summary

#--- MAIN ---#

epochs_astro = 10

time_ini = time.time()

# Set to 1 to load the weights of the encoder of the U-Net if it has already pre-trained with HI2DM.py
# It allows to explore how much astrophysical information carries the contracting part of the U-Net
# These layers are frozen and are not trained again
pretrained_encoder = 0

# Make some directories if they don't exist yet
if not os.path.exists(path+"Plots"):
    os.mkdir(path+"Plots")
if not os.path.exists(path+"Models"):
    os.mkdir(path+"Models")
if not os.path.exists(path_outputs):
    os.mkdir(path_outputs)
if not os.path.exists(path_outputs+"Outputs"+sufix):
    os.mkdir(path_outputs+"Outputs"+sufix)

# Load fields and convert to tensors
print("Loading dataset...")
inputs = load_field("dTb")
inputs = normalize_field(inputs)
tensor_x = torch.from_numpy(inputs)

# Load astropyhsical parameters, already normalized
params = np.load(path_fields+"params_sims_"+str(n_sims)+"_z_"+redshifts[0]+"_data_aug_"+str(data_aug)+".npy")
tensor_par = torch.from_numpy(params)
print("Shape data: ",tensor_x.shape,tensor_par.shape)

totaldata = utils.TensorDataset(tensor_x.float(),tensor_par.float())

# Split training and validation sets
train_loader, valid_loader, test_loader = split_datasets(totaldata)

astromodel = AstroNet()
#summary(model,(1,DIM,DIM))
lossfunc = nn.MSELoss()

if pretrained_encoder:
    print("Loading pretrained encoder weights from the U-Net")

    best_Unet_model = bestmodel
    if train_on_gpu:
        my_dict = torch.load(best_Unet_model,map_location=torch.device('cuda'))
    else:
        my_dict = torch.load(best_Unet_model,map_location=torch.device('cpu'))

    astro_state = astromodel.state_dict()

    # Copy the weights of the pretrained encoder in the correspondent layers astronet
    for (name1, param), (name2, param2) in zip(my_dict.items(), astro_state.items()):
        if name1 not in name2:
             continue
        param = param.data
        astro_state[name2].copy_(param)

    encoder = Encoder()

    for name, child in encoder.named_children():

        layer = getattr(astromodel.encoder, name)
        for param in layer.parameters():
            param.requires_grad = False

if train_on_gpu:
    astromodel.cuda()

network_total_params = sum(p.numel() for p in astromodel.parameters())
print('Total number of parameters in the model = %d'%network_total_params)
print("Data loaded. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))

# Print the memory (in GB) being used now:
process = psutil.Process()
print("Memory being used (GB):",process.memory_info().rss/1.e9)

# Train the net
if training:
    print("Learning...")
    train_losses, valid_losses = learning_loop(astromodel,train_loader,valid_loader,lossfunc,n_epochs=epochs_astro,name_model=bestmodel_astro)

    # Plot the validation/training trend
    loss_trend(train_losses,valid_losses,astro=True)


# Test the net
print("Testing...")
true_targets, predicted_targets, test_loss = testing_loop(astromodel,test_loader,lossfunc,name_model=bestmodel_astro,export_map=0)

np.savetxt(path_outputs+"AstroParams"+sufix+".dat",np.transpose([true_targets[:,0],true_targets[:,1],true_targets[:,2],predicted_targets[:,0],predicted_targets[:,1],predicted_targets[:,2]]))


# Plot true vs predicted params
param_plot(true_targets,predicted_targets,test_loss)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
