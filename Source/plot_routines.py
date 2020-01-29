#----------------------------------
# Plot routines
# Author: Pablo Villanueva Domingo
# Started 23/9/19
#----------------------------------

import numpy as np
import matplotlib.pyplot as plt
from Source.params import *
import powerbox as pbox

#--- PLOTS ---#

# Plot 2 2D slices of input, true target and predicted target respectively
def plot_slices(array1,array2,array3,index,epoch):
    # Transform to CPU arrays
    array1, array2, array3 = array1.cpu(), array2.cpu(), array3.cpu()
    for iz, z in enumerate(redshifts):
        fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(9., 3.), constrained_layout=True)
        ax_1.imshow( array1[index,iz] )
        ax_2.imshow( array2[index,iz] )
        ax_3.imshow( array3[index,iz] )
        err = (array2[index,iz]-array3[index,iz])/array2[index,iz]
        fig.suptitle("Mean relative error ={:.2f}".format(np.abs(err.mean())))
        inputlabel, targetlabel = r"$\delta T_b$", r"$\delta$"
        ax_1.set_title(inputlabel)
        ax_2.set_title(targetlabel+", true")
        ax_3.set_title(targetlabel+", predicted")
        plt.savefig(path+"Plots/2Dslices_z+"+z+"_index_"+str(index)+"_epoch_"+str(epoch)+sufix+".pdf")
        plt.close(fig)

# Compute and plot the power spectrum for the true target and predicted target
def plot_powerspectrum(true_array,predicted_array,index,epoch):
    # Transform to CPU arrays
    true_array, predicted_array = true_array.cpu(), predicted_array.cpu()
    for iz, z in enumerate(redshifts):
        fig, (ax_1) = plt.subplots(1, 1, constrained_layout=True)
        ps_true, k = pbox.tools.get_power(true_array[index,iz],DIM)
        ps_predicted, k = pbox.tools.get_power(predicted_array[index,iz],DIM)
        plt.loglog(k, ps_true,"r-",label="True")
        plt.loglog(k, ps_predicted,"b-",label="Predicted")
        err = (ps_true-ps_predicted)/ps_true
        fig.suptitle("Mean relative error ={:.2f}".format(np.abs(err.mean())))
        ax_1.set_ylabel(r"$P(k)$")
        ax_1.set_xlabel(r"$k$")
        ax_1.legend()
        plt.savefig(path+"Plots/PowerSpectrum_z+"+z+"_index_"+str(index)+"_epoch_"+str(epoch)+sufix+".pdf")
        plt.close(fig)

# Compute and plot the probability distribution function (pdf) for the true target and predicted target
def plot_pdf(true_array,predicted_array,index,epoch):
    # Transform to CPU arrays
    true_array, predicted_array = true_array.cpu(), predicted_array.cpu()
    for iz, z in enumerate(redshifts):
        fig, (ax_1) = plt.subplots(1, 1, constrained_layout=True)
        listbins = np.linspace(-3.,5.,num=30)
        hist_true, bins = np.histogram(true_array,density=True,bins=listbins)
        plt.step(bins[:-1], hist_true,"r-",label="True")
        hist_pred, bins = np.histogram(predicted_array,density=True,bins=listbins)
        plt.step(bins[:-1], hist_pred,"b-",label="Predicted")
        #err = (hist_true-hist_pred)/hist_true
        #fig.suptitle("Mean relative error ={:.2f}".format(np.abs(err.mean())))
        ax_1.set_ylabel(r"$PDF(\delta)$")
        ax_1.set_xlabel(r"$\delta$")
        ax_1.legend()
        plt.savefig(path+"Plots/PDF_z+"+z+"_index_"+str(index)+"_epoch_"+str(epoch)+sufix+".pdf")
        plt.close(fig)

# Show validation/training trend
def loss_trend(train_losses,valid_losses,test_loss):
    fig_loss, (ax_loss) = plt.subplots(1, 1)

    ax_loss.semilogy(train_losses, label='Training loss')
    ax_loss.semilogy(valid_losses, label='Validation loss')

    ax_loss.legend(frameon=False)
    ax_loss.set_title("Test loss={:.2e}".format(test_loss))
    fig_loss.savefig(path+"Plots/TrainValidLoss"+sufix+".pdf", bbox_inches='tight')
