#-----------------------------------------------------
# Some routines for plotting and computing statistics
# Author: Pablo Villanueva Domingo
# Last update 25/6/20
#-----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from Source.params import *
import powerbox as pbox
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

boxlength = 300.    # Mpc, boxlength used for simulations from 21cmFAST
suf = sufix
fontsiz = 20

#--- STATISTICS ROUTINES ---#

# Compute the mean and std of the cross correlated power spectrum of two given arrays of 2D fields
# If arrays_1==arrays_2, it gives the power spectrum.
def compute_cross_ps(arrays_1,arrays_2):

    ps_array = []
    for i, array in enumerate(arrays_1):
        ps, k = pbox.tools.get_power(array,boxlength,deltax2=arrays_2[i])
        ps_array.append(ps)
    ps_array = np.array(ps_array)
    ps_mean = np.mean(ps_array,axis=0)
    ps_std = np.std(ps_array,axis=0)
    return k, ps_mean, ps_std

# Compute the mean and std of the transfer function of two given arrays of 2D fields
def compute_transfer(arrays_pred,arrays_true):

    transf_array = []
    for i, array_pred in enumerate(arrays_pred):
        ps_pred, k = pbox.tools.get_power(array_pred,boxlength)
        ps_true, k = pbox.tools.get_power(arrays_true[i],boxlength)
        transfer = np.sqrt(ps_pred/ps_true)
        transf_array.append( transfer )
    transf_array = np.array(transf_array)
    transf_mean = np.mean(transf_array,axis=0)
    transf_std = np.std(transf_array,axis=0)
    return k, transf_mean, transf_std

# Compute the mean and std of the cross correlation coefficient of two given arrays of 2D fields
def compute_correlation(arrays_pred,arrays_true):

    corr_array = []
    for i, array_pred in enumerate(arrays_pred):
        cross_ps, k = pbox.tools.get_power(array_pred,boxlength,deltax2=arrays_true[i])
        ps_pred, k = pbox.tools.get_power(array_pred,boxlength)
        ps_true, k = pbox.tools.get_power(arrays_true[i],boxlength)
        cross_corr = cross_ps/np.sqrt( ps_pred*ps_true )
        corr_array.append(cross_corr)
    corr_array = np.array(corr_array)
    corr_mean = np.mean(corr_array,axis=0)
    corr_std = np.std(corr_array,axis=0)
    return k, corr_mean, corr_std

# Compute the PDF of an array of 2D fields
def compute_pdf(arrays,numbins=30):

    pdf_array = []
    listbins = np.linspace(-3.,4.,num=numbins)
    for array in arrays:
        hist, bins = np.histogram(array,density=True,bins=listbins)
        pdf_array.append(hist)
    pdf_array = np.array(pdf_array)
    pdf_mean = np.mean(pdf_array,axis=0)
    pdf_std = np.std(pdf_array,axis=0)
    return bins[:-1], pdf_mean, pdf_std


#--- PLOT ROUTINES ---#

# Plot 2 2D slices of input, true target and predicted target respectively
def plot_slices(array_in,array_tar,array_out,ind):

    fig, [[ ax_2, ax_3], [ax_1, ax_4]] = plt.subplots(2, 2, constrained_layout=True)
    err = np.abs(array_tar-array_out)
    v_min = np.amin(np.array([np.amin(array_tar),np.amin(array_out)]))
    v_max = np.amax(np.array([np.amax(array_tar),np.amax(array_out)]))
    cmap = "viridis"#plt.cm.viridis
    ax1 = ax_1.imshow( array_in , cmap=cmap)#, vmin=v_min, vmax=v_max)
    ax2 = ax_2.imshow( array_tar , cmap=cmap, vmin=v_min, vmax=v_max)
    ax3 = ax_3.imshow( array_out , cmap=cmap, vmin=v_min, vmax=v_max)
    ax4 = ax_4.imshow( np.log10(err) , cmap=cmap, vmin=-4., vmax=0.)

    inputlabel, targetlabel = r"$\delta T_b$", r"$\delta$"
    ax_1.set_title(inputlabel)
    ax_2.set_title(targetlabel+", true")
    ax_3.set_title(targetlabel+", predicted")
    im = ax_4.set_title(r"$log_{10}$(Error)")

    plt.colorbar(ax1, ax = ax_1, pad=0.15, anchor=(0.0, 0.5))
    plt.colorbar(ax2, ax = ax_2, pad=0.15, anchor=(0.0, 0.5))
    plt.colorbar(ax3, ax = ax_3, pad=0.15, anchor=(0.0, 0.5))
    plt.colorbar(ax4, ax = ax_4, pad=0.15, anchor=(0.0, 0.5))

    plt.savefig(path+"Plots/2Dslices_mod_ind_"+str(ind)+suf+".pdf")
    plt.close(fig)

# Plot the power spectrum for the true target and predicted target
def plot_powerspectrum(targets,outputs):

    col_corr = "k"

    fig = plt.figure(figsize=(8, 4), constrained_layout=False)
    spec = fig.add_gridspec(ncols=2, nrows=2, wspace=.25, hspace=0., left=0.1, right=0.95)
    ax_1 = fig.add_subplot(spec[:, 0])
    ax_2 = fig.add_subplot(spec[0, 1])
    ax_3 = fig.add_subplot(spec[1, 1])

    k, ps_mean_target, ps_std_target = compute_cross_ps(targets,targets)
    k, ps_mean_output, ps_std_output = compute_cross_ps(outputs,outputs)
    k, transf_mean, transf_std =  compute_transfer(outputs,targets)
    k, corr_mean, corr_std =  compute_transfer(outputs,targets)

    psstdplottrue = ax_1.fill_between(k, ps_mean_target+ps_std_target,ps_mean_target-ps_std_target,linestyle="-",color="r",alpha=0.3)
    psmeanplottrue, = ax_1.plot(k, ps_mean_target,linestyle="-",color="r",label="True",alpha=0.8)
    psstdplotpred = ax_1.fill_between(k, ps_mean_output+ps_std_output,ps_mean_output-ps_std_output,linestyle="-",color="b",alpha=0.3)
    psmeanplotpred, = ax_1.plot(k, ps_mean_output,linestyle="-",color="b",label="Predicted",alpha=0.8)

    ax_2.plot([k[0],k[-1]], [1.,1.],linestyle=":",color="k",alpha=0.1)
    ax_2.fill_between(k, transf_mean+transf_std,transf_mean-transf_std,color=col_corr,alpha=0.3)
    ax_2.plot(k, transf_mean,linestyle="-",color=col_corr,linewidth=0.75)
    ax_3.plot([k[0],k[-1]], [1.,1.],linestyle=":",color="k",alpha=0.1)
    ax_3.fill_between(k, corr_mean+corr_std,corr_mean-corr_std,color=col_corr,alpha=0.3)
    ax_3.plot(k, corr_mean,linestyle="-",color=col_corr,linewidth=0.75)

    ax_2.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_3.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_1.set_yscale("log")
    ax_1.set_xscale("log")
    ax_2.set_xscale("log")
    ax_3.set_xscale("log")
    ax_1.set_ylabel(r"$P(k) \; [Mpc^{2}]$")
    ax_2.set_ylabel(r"$T(k)$")
    ax_3.set_ylabel(r"$r(k)$")
    ax_1.set_xlabel(r"$k \; [Mpc^{-1}]$")
    ax_3.set_xlabel(r"$k \; [Mpc^{-1}]$")
    ax_2.set_xticks([])

    kNyquist = DIM*np.pi/boxlength # = 2.1, k[-1]=2.9
    ax_1.set_xlim([k[0],kNyquist])
    ax_2.set_xlim([k[0],kNyquist])
    ax_3.set_xlim([k[0],kNyquist])
    ax_2.set_ylim([0.85,1.1])
    ax_3.set_ylim([0.85,1.1])
    ax_1.legend()
    err_ps = (np.abs(ps_mean_target-ps_mean_output))/ps_mean_target

    plt.savefig(path+"Plots/PowerSpectrum"+suf+".pdf")
    plt.close(fig)

# Plot the probability distribution function (pdf) for the true target and predicted target
def plot_pdf(targets,outputs):

    fig, (ax_1) = plt.subplots(1, 1, constrained_layout=True)

    bins, pdf_mean_target, pdf_std_target = compute_pdf(targets,numbins=15)
    bins, pdf_mean_output, pdf_std_output = compute_pdf(outputs,numbins=15)

    # Interpolate the histograms
    finebins = np.linspace(bins[0],bins[-1],num=200)
    kindint = "quadratic"
    int_mean_tar, int_std_tar = interp1d(bins, pdf_mean_target,kind=kindint), interp1d(bins, pdf_std_target,kind=kindint)
    int_mean_out, int_std_out = interp1d(bins, pdf_mean_output,kind=kindint), interp1d(bins, pdf_std_output,kind=kindint)

    ax_1.plot(finebins, int_mean_tar(finebins),"r-",label="True")
    ax_1.fill_between(finebins, int_mean_tar(finebins)+int_std_tar(finebins), int_mean_tar(finebins)-int_std_tar(finebins), color="r", alpha=0.3)
    ax_1.plot(finebins, int_mean_out(finebins),"b-",label="Predicted")
    ax_1.fill_between(finebins, int_mean_out(finebins)+int_std_out(finebins), int_mean_out(finebins)-int_std_out(finebins), color="b", alpha=0.3)

    ax_1.set_ylabel(r"$PDF(\delta)$",fontsize=fontsiz)
    ax_1.set_xlabel(r"$\delta$",fontsize=fontsiz)
    ax_1.set_xlim([bins[0],bins[-1]])
    ax_1.legend(fontsize=fontsiz)
    plt.savefig(path+"Plots/PDF"+suf+".pdf")
    plt.close(fig)

# Show validation/training trend
def loss_trend(train_losses,valid_losses,astro=False):

    fig_loss, (ax_loss) = plt.subplots(1, 1)
    epochs = range(1,len(train_losses)+1)
    ax_loss.semilogy(epochs, train_losses, label='Training loss', color="cyan")
    ax_loss.semilogy(epochs, valid_losses, label='Validation loss', color="purple")

    ax_loss.xaxis.set_minor_locator(MultipleLocator(2))
    ax_loss.xaxis.set_major_locator(MultipleLocator(4))
    ax_loss.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_loss.legend(frameon=False, fontsize=fontsiz)
    ax_loss.set_ylabel("Loss",fontsize=fontsiz)
    ax_loss.set_xlabel("Epochs",fontsize=fontsiz)
    ax_loss.set_xlim([epochs[0],epochs[-1]])

    if astro==True:
        fig_loss.savefig(path+"Plots/TrainValidLoss_astro"+suf+".pdf", bbox_inches='tight')
    else:
        fig_loss.savefig(path+"Plots/TrainValidLoss"+suf+".pdf", bbox_inches='tight')
    plt.close(fig_loss)

# Show true vs predicted astrophysical parameters
def param_plot(true_params,predicted_params,test_loss):

    fig_par, (ax_m, ax_lx, ax_ng) = plt.subplots(1, 3, figsize=(9., 3.), constrained_layout=True)
    fig_par.suptitle("$R^2$={:.2f}".format(r2_score(true_params,predicted_params)))

    pointsize = 1
    ax_m.plot(true_params[:,0], predicted_params[:,0], "bo", markersize=pointsize)
    ax_lx.plot(true_params[:,1], predicted_params[:,1], "bo", markersize=pointsize)
    ax_ng.plot(true_params[:,2], predicted_params[:,2], "bo", markersize=pointsize)

    ax_m.plot(true_params[:,0], true_params[:,0], "r-")
    ax_lx.plot(true_params[:,1], true_params[:,1], "r-")
    ax_ng.plot(true_params[:,2], true_params[:,2], "r-")

    ax_m.set_xlabel(r"$M_{turn}$, True")
    ax_lx.set_xlabel(r"$L_{X}$, True")
    ax_ng.set_xlabel(r"$N_{\gamma}$, True")

    ax_m.set_ylabel(r"$M_{turn}$, Predicted")
    ax_lx.set_ylabel(r"$L_{X}$, Predicted")
    ax_ng.set_ylabel(r"$N_{\gamma}$, Predicted")

    fig_par.savefig(path+"Plots/ParameterPrediction"+suf+".png", bbox_inches='tight', dpi=300)
    plt.close(fig_par)
