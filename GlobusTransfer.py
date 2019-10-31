import os

numsims = 50
redshifts = ["010.16","015.78","020.18"]

# Globus locations
ep1="85b4194a-daf1-11e9-87d3-025f0df9da94"
ep2="a50b79f6-daf5-11e9-b5de-0ef30f6b83a8"

# Transfer files
def transfer(origin,destination):
    if not os.path.exists("/Users/omena/"+destination):
        os.system("globus transfer "+ep1+":~/"+origin+" "+ep2+":~/"+destination)

# Main
for i in range(1,numsims+1):
    if i % 5 == 0: print("Simulation",i)
    path_orig = "DM_2_HI/Simulation_"+str(i)+"/Boxes/"
    path_dest = "Downloads/21_DeepLearning/Files_DM2HI/Simulation_"+str(i)+"/"
    for z in redshifts:
        dTbfile = "dTb_z"+z
        deltafile = "updated_smoothed_deltax_z"+z+"_200_300Mpc"
        transfer( path_orig+dTbfile, path_dest+dTbfile )
        transfer( path_orig+deltafile, path_dest+deltafile )
