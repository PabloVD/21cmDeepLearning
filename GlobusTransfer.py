# Code for importing simulation files from a Globus repository.
# You may need a Globus account for that.
# Last modification: 6/11/19

from Source.params import *

# Firstly, initialize Globus app

# Globus endpoint locations.
# Leave ep1 the same, it is the source where the simulations are transferred from.
# Change ep2 accordingly to your endpoint. You can find the "Endpoint UUID" in the Globus web, in "Endpoints".
ep1="85b4194a-daf1-11e9-87d3-025f0df9da94"
ep2="a50b79f6-daf5-11e9-b5de-0ef30f6b83a8"

# Transfer files function
def transfer(origin,destination):
    if not os.path.exists(destination):
        os.system("globus transfer "+ep1+":"+origin+" "+ep2+":"+destination)
    else:   print(destination+" does already exist.")

# Main loop
for i in range(1,n_sims+1):
    if i % 5 == 0: print("Simulation",i)
    path_orig = "DM_2_HI/Simulation_"+str(i)+"/Boxes/"
    path_dest = path+"Files_DM2HI/Simulation_"+str(i)+"/"
    for z in redshifts:
        dTbfile = "dTb_z"+z
        deltafile = "updated_smoothed_deltax_z"+z+"_200_300Mpc"
        transfer( path_orig+dTbfile, path_dest+dTbfile )
        transfer( path_orig+deltafile, path_dest+deltafile )
