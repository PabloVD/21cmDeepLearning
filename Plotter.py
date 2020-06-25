#-------------------------------------------------------------
# Driver for plotting several statistics
# Author: Pablo Villanueva Domingo
# Last update: 25/6/20
#-------------------------------------------------------------

import time, datetime
import glob
from Source.plot_routines import *

time_ini = time.time()

# Load the output files of the trained network

target_files = sorted(glob.glob(path_outputs+"Outputs"+suf+"/slice_target*"))
output_files = sorted(glob.glob(path_outputs+"Outputs"+suf+"/slice_output*"))

# Take a reduced set of files
num_testfiles = 100
target_files = target_files[:num_testfiles]
output_files = output_files[:num_testfiles]

numfiles = len(target_files)
if numfiles!=len(output_files):
    print("Not the same number of targets and outputs!")

targets = np.empty((numfiles, DIM, DIM), dtype=np.float32)
outputs = np.empty((numfiles, DIM, DIM), dtype=np.float32)

for i, file in enumerate(target_files):
    targets[i] = np.load(file)
for i, file in enumerate(output_files):
    outputs[i] = np.load(file)

# Plot some 2D slices samples

for ind in [0,30,60,90]:
    file_input = np.load(path_outputs+"Outputs"+suf+"/slice_input_"+str(ind)+".npy")
    file_target = np.load(path_outputs+"Outputs"+suf+"/slice_target_"+str(ind)+".npy")
    file_output = np.load(path_outputs+"Outputs"+suf+"/slice_output_"+str(ind)+".npy")
    plot_slices(file_input,file_target,file_output,ind)

# Train and validation error plot_pdf

losses = np.loadtxt(path_outputs+"Losses"+suf+".dat",unpack=True)
loss_trend(losses[0],losses[1])

# Power spectrum

plot_powerspectrum(targets,outputs)

# PDF

plot_pdf(targets,outputs)


print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
