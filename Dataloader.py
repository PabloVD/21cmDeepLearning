#----------------------------
# Preprocess data code
# Transform from binary files to numpy arrays and save them in npy format
# Author: Pablo Villanueva Domingo
# Started 23/9/19
#----------------------------

import numpy as np
import time, os, glob
from Source.params import *

time_ini = time.time()

path_globus = "Files_DM2HI/"#"GlobusFiles/"
path_fields = "Fields/"

# Read parameters file and normalize the paramters to (0,1)
def get_params():

    paramfile = "GlobusFiles/params_simulation_mturn_lumx_ngamma.txt"
    paramstab = np.loadtxt(paramfile)

    p_max, p_min = [], []
    for i in range(1,4):
        p_max.append(np.amax(paramstab[:,i]))
        p_min.append(np.amin(paramstab[:,i]))


    sims = []
    for numsim in range(1,n_sims+1):

        filename = "GlobusFiles/Simulation_"+str(numsim)+"/dTb_z015.78"
        if os.path.exists(filename):

            par = paramstab[numsim-1][1:4]
            par = np.array([ (par[i]-p_min[i])/(p_max[i] - p_min[i]) for i in range(3) ])
            #sims.append(par)       # 1 time, without data augmentation
            for i in range(0,8):
                sims.append(par)    # 8 times, for data augmentation

    np.save("Inputfiles/params.npy",np.array(sims))


def get_array(filename,outfile):

    # Check if the fields are already converted
    outfiles = glob.glob(outfile+"*")
    if len(outfiles)<10:
        if os.path.exists(filename):

            data = np.fromfile(filename, dtype=np.float32)
            delta = np.zeros((DIM,DIM,DIM))

            n = 0
            for i in range(DIM):
                for j in range(DIM):
                    for k in range(DIM):
                        delta[i,j,k] = data[n]
                        n+=1

            for l in range(0,DIM,20):   # take 10 slices separated by 20 bins in each dimension

                deltaprime_x = delta[l,:,:]
                deltaprime_y = delta[:,l,:]
                deltaprime_z = delta[:,:,l]

                np.save(outfile+"_x_bin_"+str(l)+".npy",deltaprime_x)
                np.save(outfile+"_y_bin_"+str(l)+".npy",deltaprime_y)
                np.save(outfile+"_z_bin_"+str(l)+".npy",deltaprime_z)

        else: print(filename," doesn't exist!")

# Read dTb files
def get_data():

    sims = []
    for numsim in range(1,n_sims+1):

        if numsim % 5 == 0: print("Simulation",numsim)

        for z in redshifts:

            dTb_filename = path_globus+"Simulation_"+str(numsim)+"/dTb_z"+z
            delta_filename = path_globus+"Simulation_"+str(numsim)+"/updated_smoothed_deltax_z"+z+"_200_300Mpc"
            dTb_outfile = path_fields+"Simulation_"+str(numsim)+"_dTb_z"+z
            delta_outfile = path_fields+"Simulation_"+str(numsim)+"_delta_z"+z

            get_array(dTb_filename,dTb_outfile)
            get_array(delta_filename,delta_outfile)


#get_params()
get_data()

print("Minutes elapsed:",(time.time()-time_ini)/60.)
