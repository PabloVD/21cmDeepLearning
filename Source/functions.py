#----------------------------------
# Useful functions and routines
# Author: Pablo Villanueva Domingo
# Last update: 25/6/20
#----------------------------------

import random, glob
import numpy as np
import torch
import torch.utils.data as utils
from torch.optim import Adam, lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from Source.params import *

# Set random seed
random_seed = 123
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Use CUDA GPUs if available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('\nCUDA is available! Training on GPU.')
    device = torch.device('cuda')
else:
    print('\nCUDA is not available. Training on CPU.')
    device = torch.device('cpu')


#--- MANIPULATING DATA ---#

# Load all the fields of type fieldtype (namely, 21 cm map or matter density field)
def load_field(fieldtype):

    # Count number of useful simulations (some of the simulations may not be complete and not used)
    numsimsreal = 0
    for numsim in range(1,n_sims+1):
        filename = path_fields+"Simulation_"+str(numsim)+"/delta_z"+redshifts[0]+"_bin_0.npy"
        if os.path.exists(filename):
            numsimsreal+=1
    numsimsreal*=20     # 20 slices per simulation

    if data_aug:    numsimsreal*=8  # 8 possible transformations
    ind = 0
    fields = np.empty((numsimsreal, 1, DIM, DIM), dtype=np.float32)

    for numsim in range(1,n_sims+1):

        # Pass if there are no arrays of these simulation
        arrayssim = glob.glob(path_fields+"Simulation_"+str(numsim)+"/*")
        if len(arrayssim)==0:
            continue
            print("No fields in simulation",numsim)

        for bin in range(0,DIM,10):
            for z in redshifts:
                filename = path_fields+"Simulation_"+str(numsim)+"/"+fieldtype+"_z"+z+"_bin_"+str(bin)+".npy"
                if os.path.exists(filename):
                    field = np.load(filename)
                    if data_aug:    # If data augmentation, employ 8 possible rigid transformations in 2D
                        for i in range(0,4):    # original + 3 rotations
                            fields[ind+i] = np.rot90(field,i).reshape(1, DIM, DIM)
                        for i in range(4,8):    # inversion of the original + 3 rotations
                            fields[ind+i] = np.rot90(np.flip(field,0),i).reshape(1, DIM, DIM)
                        ind+=8
                    else:
                        fields[ind] = field.reshape(1, DIM, DIM)
                        ind+=1
                else:   print(filename+" doesn't exist.")#"""

    return fields

# Normalize the inputs
def normalize_field(fields):
    mean, std = fields.mean(), fields.std()
    for sim in fields:
        for i, z in enumerate(redshifts):
            sim[i] = (sim[i] - mean)/std
    return fields

# Split training and validation sets
def split_datasets(totaldata):
    num_train = len(totaldata)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))
    train_idx, valid_idx, test_idx = indices[split_test:], indices[:split_valid], indices[split_valid:split_test]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create loaders
    train_loader = utils.DataLoader(totaldata, batch_size=batch_size, sampler=train_sampler)
    valid_loader = utils.DataLoader(totaldata, batch_size=batch_size, sampler=valid_sampler)
    test_loader = utils.DataLoader(totaldata, batch_size=batch_size, sampler=test_sampler)

    return train_loader, valid_loader, test_loader


#--- MACHINE LEARNING LOOPS ---#

# Loop where training and validation are perfomed
def learning_loop(model,train_loader,valid_loader,lossfunc,n_epochs,name_model=bestmodel):

    # Adam optimizer
    # The filter ensures that only trainable layers are updated
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay)

    # Learning rate scheduler, suppresses by 0.1 the learning rate after n_epochs/4 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(n_epochs/4), gamma=0.1)

    train_losses, valid_losses = [], []
    valid_loss_min = np.Inf

    # Start learning!
    for epoch in range(1,n_epochs+1):
        train_loss, valid_loss = 0., 0.

        # Training loop
        model.train()
        for input, target in train_loader:
            if train_on_gpu:
                input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = lossfunc(output, target)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        # Validation loop
        with torch.no_grad():
            model.eval()
            for input, target in valid_loader:
                if train_on_gpu:
                    input, target = input.cuda(), target.cuda()
                output = model(input)
                loss = lossfunc(output, target)
                valid_loss+=loss.item()

        scheduler.step()

        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        train_losses.append(train_loss), valid_losses.append(valid_loss)

        print("Epoch=",epoch,", Train loss={:.2e}, Validation loss={:.2e}".format(train_loss,valid_loss) )

        # Save model if it has improved
        if valid_loss <= valid_loss_min:
            print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), name_model)
            valid_loss_min = valid_loss

        # Write loss to a file in real time
        f = open("lossfile"+sufix+".dat", 'a')
        f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
        f.close()

    np.savetxt(path_outputs+"Losses"+sufix+".dat",np.transpose([np.array(train_losses),np.array(valid_losses)]))

    return train_losses, valid_losses

# Loop for testing the trained network
# Use export_map=1 for exporting samples of 2D maps. Only with the U-Net, not with the astro net
def testing_loop(model,test_loader,lossfunc,name_model=bestmodel,export_map=1):

    # Load the best model
    state_dict = torch.load(name_model, map_location=device)
    model.load_state_dict(state_dict)

    # Test the model
    ind_tar, ind_out = 0, 0
    test_loss = 0.0
    true_target, predicted_target = [], []
    with torch.no_grad():
        model.eval()
        for input, target in test_loader:
            if train_on_gpu:
                input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = lossfunc(output, target)
            test_loss+=loss.item()

            # Store some outputs and export some sample maps
            if export_map:      # Only export one input per batch (just for plots)
                np.save(path_outputs+"Outputs"+sufix+"/slice_input_"+str(ind_tar),input[0].cpu().reshape(200,200))
            for tar in target:
                true_target.append(tar.cpu().numpy())
                if export_map:
                    np.save(path_outputs+"Outputs"+sufix+"/slice_target_"+str(ind_tar),tar.cpu().reshape(200,200))
                ind_tar+=1
            for out in output:
                predicted_target.append(out.cpu().numpy())
                if export_map:
                    np.save(path_outputs+"Outputs"+sufix+"/slice_output_"+str(ind_out),out.cpu().reshape(200,200))
                ind_out+=1

    true_target, predicted_target = np.array(true_target), np.array(predicted_target)
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.2e}\n'.format(test_loss))
    return true_target, predicted_target, test_loss
