#--------------------------------------------------------------------------
# Preprocess data code
# Transform from binary files to numpy arrays and store them in npy format
# Author: Pablo Villanueva Domingo
# Last update: 24/6/20
#--------------------------------------------------------------------------

import numpy as np
import time, os, glob
from Source.params import *

time_ini = time.time()
zz = redshifts
nsims = n_sims
sim_ini = 1
sep = 10    # separation between bins
num_sli = int(DIM/sep)   # number of slices per simulation


if not os.path.exists(path_fields):
    os.system("mkdir "+path_fields)

#--- ROUTINES ---#

# Read the binary file and convert it to a numpy array stored in a .npy file
def get_array(filename,outfile):

    # Check if the fields are already converted
    outfiles = glob.glob(outfile+"*")
    if len(outfiles)<num_sli:
        if os.path.exists(filename):

            data = np.fromfile(filename, dtype=np.float32)
            delta = np.zeros((DIM,DIM,DIM))

            n = 0
            for i in range(DIM):
                for j in range(DIM):
                    for k in range(DIM):
                        delta[i,j,k] = data[n]
                        n+=1

            # take 20 slices separated by 10 bins in each dimension
            # take a width of 4 bins, project them to a 2D slice
            for l in range(0,DIM,sep):

                deltaprime_z = delta[:,:,l:l+4]
                deltaprime_z = deltaprime_z.sum(axis=2)

                np.save(outfile+"_bin_"+str(l)+".npy",deltaprime_z)

        else: print(filename," doesn't exist!")

# Read dTb and delta files
def get_data():

    for numsim in range(sim_ini,nsims+1):

        if numsim % 5 == 0: print("Simulation",numsim)

        if not os.path.exists(path_fields+"Simulation_"+str(numsim)):
            os.system("mkdir "+path_fields+"Simulation_"+str(numsim))

        for z in zz:
            dTb_loc = glob.glob(path_simulations+"Simulation_"+str(numsim)+"/Boxes/delta_T_v3_z"+z+"*")
            # ensure that the file exists
            if len(dTb_loc)==1:
                dTb_filename = dTb_loc[0]
                delta_filename = path_simulations+"Simulation_"+str(numsim)+"/Boxes/updated_smoothed_deltax_z"+z+"_200_300Mpc"
                dTb_outfile = path_fields+"Simulation_"+str(numsim)+"/dTb_z"+z
                delta_outfile = path_fields+"Simulation_"+str(numsim)+"/delta_z"+z

                get_array(dTb_filename,dTb_outfile)
                get_array(delta_filename,delta_outfile)
            else:
                print("dTb file in Simulation "+str(numsim)+" doesn't exist!")


# This function ensures that we have the same number of dTb arrays for the correspondent delta arrays (just a sanity check, some simulations may not produce both)
# Removes the arrays of the simulations which present this kind of problem
def check_arrays():

    for numsim in range(sim_ini,nsims+1):
        for z in zz:
            dTb_outfile = path_fields+"Simulation_"+str(numsim)+"/dTb_z"+z
            delta_outfile = path_fields+"Simulation_"+str(numsim)+"/delta_z"+z

            dTb_outfiles = glob.glob(dTb_outfile+"*")
            delta_outfiles = glob.glob(delta_outfile+"*")

            if len(dTb_outfiles)!=len(delta_outfiles):
                print("Simulation_"+str(numsim)+" does not have the same number of delta arrays and dTb arrays. Removing them...")
                for file in dTb_outfiles:
                    os.system("rm "+file)
                for file in delta_outfiles:
                    os.system("rm "+file)


# Read parameters file and normalize them to (0,1)
def get_params():

    paramfile = path_simulations+"params_simulation_mturn_lumx_ngamma.txt"
    paramstab = np.loadtxt(paramfile)

    p_max, p_min = [], []
    for i in range(1,4):
        p_max.append(np.amax(paramstab[:,i]))
        p_min.append(np.amin(paramstab[:,i]))

    for z in redshifts:

        sims = []
        for numsim in range(1,n_sims+1):

            dTb_outfile = path_fields+"Simulation_"+str(numsim)+"/dTb_z"+z
            delta_outfile = path_fields+"Simulation_"+str(numsim)+"/delta_z"+z

            dTb_outfiles = glob.glob(dTb_outfile+"*")
            delta_outfiles = glob.glob(delta_outfile+"*")

            # take only the simulations considered for the Unet, which have 20 slices per 3D box
            if len(dTb_outfiles)==num_sli and len(delta_outfiles)==num_sli:

                par = paramstab[numsim-1][1:4]
                par = np.array([ (par[i]-p_min[i])/(p_max[i] - p_min[i]) for i in range(3) ])

                for n in range(num_sli): # 30 slices per 3D box

                    if data_aug:
                        for i in range(0,8):
                            sims.append(par)    # 8 times, for data augmentation
                    else:
                        sims.append(par)       # 1 time, without data augmentation

        np.save(path_fields+"params_sims_"+str(nsims)+"_z_"+z+"_data_aug_"+str(data_aug)+".npy",np.array(sims))


#--- MAIN ---#

get_data()

check_arrays()

get_params()

print("Minutes elapsed:",(time.time()-time_ini)/60.)
