# Code for importing simulation files from a Globus repository.
# You may need a Globus account for that.
# Last modification: 6/11/19

from Source.params import *

# Firstly, initialize Globus app

nsims = 1000 #n_sims
zz = ["020.18"]#redshifts

# Globus endpoint locations.
# Leave ep1 the same, it is the source where the simulations are transferred from.
# Change ep2 accordingly to your endpoint. You can find the "Endpoint UUID" in the Globus web, in "Endpoints".
ep1="85b4194a-daf1-11e9-87d3-025f0df9da94"
ep2="a9df83d2-42f0-11e6-80cf-22000b1701d1"  # princeton tigress
#ep2="d10905d0-3709-11ea-b961-0e16720bb42f", # pdomingo in tiger

# Transfer files function
def transfer(origin,destination):
    if not os.path.exists(destination):
        os.system("globus transfer "+ep1+":"+origin+" "+ep2+":"+destination)
    else:   print(destination+" does already exist.")

def substract(a, b):
    return "".join(a.rsplit(b))

# Main loop
for i in range(1,nsims+1):
    if i % 5 == 0: print("Simulation",i)
    path_orig = "/Parameter_estimatior/Simulation_"+str(i)+"/Boxes/"
    path_dest = path_globus+"Simulation_"+str(i)+"/"

    os.system( "globus ls "+ep1+":"+path_orig+" > outs.txt" )
    with open("outs.txt","r") as f:
        lines = f.readlines()

    for z in zz:

        origin = ""
        for j in range(len(lines)):
            if lines[j].startswith("delta_T_v3_z"+z):
                origin = path_orig+substract(lines[j], "\n")
        if origin=="":
            print("dTb in sim "+str(i)+", z="+z+" not available")
            continue

        dTbfile = "dTb_z"+z
        deltafile = "updated_smoothed_deltax_z"+z+"_200_300Mpc"
        transfer( origin, path_dest+dTbfile )
        transfer( path_orig+deltafile, path_dest+deltafile )
