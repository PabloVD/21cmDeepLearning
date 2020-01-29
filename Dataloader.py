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

nsims = 1000

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

# Read dTb and delta files
def get_data():

    sims = []
    for numsim in range(1,nsims+1):

        if numsim % 5 == 0: print("Simulation",numsim)

        for z in redshifts:

            dTb_filename = path_globus+"Simulation_"+str(numsim)+"/dTb_z"+z
            delta_filename = path_globus+"Simulation_"+str(numsim)+"/updated_smoothed_deltax_z"+z+"_200_300Mpc"
            dTb_outfile = path_fields+"Simulation_"+str(numsim)+"_dTb_z"+z
            delta_outfile = path_fields+"Simulation_"+str(numsim)+"_delta_z"+z

            get_array(dTb_filename,dTb_outfile)
            get_array(delta_filename,delta_outfile)

# This function ensures that we have the same number of dTb arrays for the correspondent delta arrays (some simulations did not produced both)
# Removes the arrays of the simulations which present this kind of problem
def check_arrays():
    for numsim in range(1,nsims+1):
        for z in redshifts:
            dTb_outfile = path_fields+"Simulation_"+str(numsim)+"_dTb_z"+z
            delta_outfile = path_fields+"Simulation_"+str(numsim)+"_delta_z"+z

            dTb_outfiles = glob.glob(dTb_outfile+"*")
            delta_outfiles = glob.glob(delta_outfile+"*")

            if len(dTb_outfiles)!=len(delta_outfiles):
                print("Simulation_"+str(numsim)+" does not have the same number of delta arrays and dTb arrays. Removing them...")
                for file in dTb_outfiles:
                    os.system("rm "+file)
                for file in delta_outfiles:
                    os.system("rm "+file)

get_data()

check_arrays()

print("Minutes elapsed:",(time.time()-time_ini)/60.)
