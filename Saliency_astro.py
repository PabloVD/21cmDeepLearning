#-------------------------------------------------------------------------------------
# Compute the saliency maps for the AstroNet using Vainilla Gradient (arXiv:1312.6034)
# Author: Pablo Villanueva Domingo
# Last update: 25/6/20
#-------------------------------------------------------------------------------------

import time, datetime
from Source.functions import *
from Source.nets import AstroNet
import matplotlib.pyplot as plt

time_ini = time.time()

# Set to 1 for producing a saliency map for each astro parameter
# Set to 0 for only the saliency map for the first parameter
all_maps = 1

# Maximum loss to be considered for computing the saliency map
max_loss = 0.2

# Color maps for plots
cmap_21 = "viridis"
cmap_points = "Reds"

# Threshold for plot points of the saliency map
threshold = 5.

# Computes the saliency map given an image and its astrophysical parameters
def saliency_map(X,Ytrue,nsim):

    mean, std = X.mean(), X.std()
    X = (X - mean)/std

    X = torch.tensor(X,dtype=torch.float32)
    Xmod = X.view((1,1,X.shape[0],X.shape[1]))

    # Load the astro model
    bestmodel_name = bestmodel_astro
    model = AstroNet()
    state_dict = torch.load(bestmodel_name, map_location=device)
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    if train_on_gpu:
        Xmod = Xmod.cuda()
        model.cuda()

    model.eval()
    Xmod.requires_grad_()

    # Feed the model with the image and compute the loss
    Yout = model(Xmod)
    Ytrue = torch.tensor(Ytrue,dtype=torch.float32).view(1,3)
    loss = np.sqrt((Yout.detach()-Ytrue)**2.).mean()

    # Only images with low loss are considered
    if loss<max_loss:

        print("Simulation:",nsim,",Loss:",float(loss), ", Parameters:",*Ytrue)

        # Compute the gradients wrt the input image
        sals = []
        for y in Yout.view(3):
            y.backward(retain_graph=True)
            saliency = Xmod.grad.data.abs()
            saliency = saliency.cpu().numpy().reshape(X.shape)
            sals.append(saliency)

        if all_maps:
            fig, [ax2, ax3, ax4] = plt.subplots(1,3)
            for i, ax in enumerate([ax2, ax3, ax4]):
                ax.imshow(X, cmap=cmap_21)
                sali = sals[i]
                # Show only larger points of the saliency map, above mean*threshold
                sali = np.ma.masked_where( (sali/sali.mean() < threshold), sali)
                ax.imshow(np.log10(sali), cmap=cmap_points)
                ax.set_axis_off()

            fontsize = 8
            ax2.set_title(r"Saliency map $M_{turn}$",fontsize=fontsize)
            ax3.set_title(r"Saliency map $L_{X}$",fontsize=fontsize)
            ax4.set_title(r"Saliency map $N_{\gamma}$",fontsize=fontsize)

        else:
            fig, ax1 = plt.subplots(1,1)
            ax_1 = ax1.imshow(X, cmap=cmap_21)
            sali = sals[0]
            # Show only larger points of the saliency map, above mean*threshold
            sali = np.ma.masked_where( (sali/sali.mean() < threshold), sali)
            ax1.imshow(np.log10(sali), cmap=cmap_points)
            ax1.set_axis_off()

        plt.savefig("Plots/Saliency_map_"+sufix+"_sim_"+str(nsim)+"_all_params_"+str(all_maps)+".pdf", bbox_inches='tight')
        plt.close(fig)


#--- MAIN ---#

z = redshifts[0]
nini = 1
nsims = n_sims
set_sims = range(nini,nsims+1)

# Loop over several simulations
for numsim in set_sims:
    bin = 20    # Choose a particular bin of the 3D box
    dTbfile = path_fields+"Simulation_"+str(numsim)+"/dTb_z"+z+"_bin_"+str(bin)+".npy"
    params = np.load(path_fields+"params_sims_"+str(n_sims)+"_z_"+redshifts[0]+"_data_aug_"+str(data_aug)+".npy")
    Ytrue = params[(numsim-1)*8*20] # Find the correspondent astrophysical parameter. 8 and 20 stand for the data augmentation and the 20 bins per simulation
    field = np.load(dTbfile)
    saliency_map(field,Ytrue,numsim)


print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
