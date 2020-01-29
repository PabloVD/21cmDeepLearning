#----------------------------------
# Useful functions and routines
# Author: Pablo Villanueva Domingo
# Started 23/9/19
#----------------------------------

import random, glob
import torch
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
#from sklearn.metrics import r2_score
from Source.plot_routines import *

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
#if not train_on_gpu:    print('CUDA is not available.  Training on CPU ...')
#else:   print('CUDA is available!  Training on GPU ...')

# use GPUs if available
if train_on_gpu:
    print('CUDA is available! Training on GPU.')
    device = torch.device('cuda')
else:
    print('CUDA is not available. Training on CPU.')
    device = torch.device('cpu')

#--- MANIPULATING DATA ---#

# Rotates and flips in all the 8 possible ways for a number of channels n_channels
def data_augmentation(array,n_channels):
    transf = []
    for i in range(0,4):
        transfz, transfzflip = [], []
        for j in range(n_channels):
            transfz.append(np.rot90(array[j],i))
            transfzflip.append(np.rot90(array[j],i))
        transf.append(transfz)
        transf.append(transfzflip)
    return transf

# Load a field of type fieldtype
def load_field(fieldtype):
    fields = []
    for numsim in range(1,n_sims+1):

        arrayssim = glob.glob(path_fields+"Simulation_"+str(numsim)+"*")
        if len(arrayssim)==0: continue # Pass if there are no arrays of these simulation

        for bin in range(0,DIM,20):
            for coord in ["x","y","z"]:
                fieldsz = []
                for z in redshifts:
                    filename = path_fields+"Simulation_"+str(numsim)+"_"+fieldtype+"_z"+z+"_"+coord+"_bin_"+str(bin)+".npy"
                    if os.path.exists(filename):
                        field = np.load(filename)
                        fieldsz.append(field)
                    else:   print(filename+" doesn't exist.")
                if data_aug:
                    fields.extend( data_augmentation(fieldsz,n_channels) )    # with data augmentation
                else:
                    fields.append( fieldsz )                       # without data augmentation

    return np.array(fields)

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


#--- ML LOOPS ---#

# Loop where training and validation are computed
def learning_loop(model,train_loader,valid_loader,lossfunc,n_epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses, valid_losses = [], []
    valid_loss_min = np.Inf

    #if train_on_gpu:
    #    model.cuda()

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

                #if plot_sli: plot_slices(input,target,output,0,epoch)
                #if plot_pow: plot_powerspectrum(target,output,0,epoch)
                #if plot_pdf: plot_pdf(target,output,0,epoch)

        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        train_losses.append(train_loss), valid_losses.append(valid_loss)

        print("Epoch=",epoch,", Train loss={:.2e}, Validation loss={:.2e}".format(train_loss,valid_loss) )

        # Save model if it has improved
        if valid_loss <= valid_loss_min:
            print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), bestmodel)
            valid_loss_min = valid_loss

    np.savetxt(path+"Losses"+sufix+".dat",np.transpose([np.array(train_losses),np.array(valid_losses)]))

    return train_losses, valid_losses

# Loop where testing is computed
def testing_loop(model,test_loader,lossfunc):
    # Load the best model
    state_dict = torch.load(bestmodel)
    model.load_state_dict(state_dict)

    # Test the model
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
            for tar in target:
                true_target.append(tar.cpu().numpy())
            for tar in output:
                predicted_target.append(tar.cpu().numpy())

            #if plot_sli: plot_slices(input,target,output,0,"test")
            #if plot_pow: plot_powerspectrum(target,output,0,"test")
            #if plot_pdf: plot_pdf(target,output,0,"test")

        if plot_sli: plot_slices(input,target,output,0,"test")
        if plot_pow: plot_powerspectrum(target,output,0,"test")
        if plot_pdf: plot_pdf(target,output,0,"test")

    true_target, predicted_target = np.array(true_target), np.array(predicted_target)
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.2e}\n'.format(test_loss))
    return true_target, predicted_target, test_loss
