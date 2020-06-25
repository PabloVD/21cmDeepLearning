#--------------------------------------------
# Convolutional networks (U-Net and AstroNet)
# Author: Pablo Villanueva Domingo
# Last update: 23/6/20
#--------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Channel sizes
chn1 = 64
chn2 = chn1*2
chn3 = chn2*2
chn4 = chn3*2

#--- BLOCKS ---#

# Convolutional block: Conv2d + BN + ReLu (unit layer in the paper)
# Output dimension Df = (Di -K +2P)/S +1 (K: kernel, P: padding, S: stride)
# With default hyperparameters, Df = Di
def miniblock(in_channels, out_channels, stride=1, kernel_size=3):
    block = nn.Sequential(
                nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    return block

# Contracting block, formed by 2 unit layers or miniblocks
def contracting_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            miniblock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            miniblock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
            )
    return block

# Expansive block, formed by 2 unit layers or miniblocks, followed by a transpose convolution for upsampling
def expansive_block(in_channels, mid_channel, out_channels, kernel_size=3):
    block = nn.Sequential(
            miniblock(in_channels=in_channels, out_channels=mid_channel, kernel_size=kernel_size),
            miniblock(in_channels=mid_channel, out_channels=mid_channel, kernel_size=kernel_size),
            nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
    return  block

# Final block, formed by 2 miniblocks plus a final convolutional layer
def final_block(in_channels, mid_channel, out_channels, kernel_size=3):
    block = nn.Sequential(
            miniblock(in_channels=in_channels, out_channels=mid_channel, kernel_size=kernel_size),
            miniblock(in_channels=mid_channel, out_channels=mid_channel, kernel_size=kernel_size),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
            )
    return  block

# Crop the layer from a contracting block and concatenate it with the expansive block of the same level
# The resulting layer has the channels of the upsampled plus the bypass layers (from the contracting block)
def crop_and_concat(upsampled, bypass, crop=False):
    if crop:    # Only needed if bypass size is different than upsampled size
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

#--- NETWORKS ---#

# U-Net architecture
# Based on the implementation from https://github.com/Hsankesara/DeepResearch/tree/master/UNet
class UNet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        #Encoder
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=chn1)
        self.conv_encode2 = contracting_block(chn1, chn2)
        self.conv_encode3 = contracting_block(chn2, chn3)

        # Decoder
        self.bottleneck = expansive_block(chn3, chn4, chn3)
        self.conv_decode3 = expansive_block(chn4, chn3, chn2)
        self.conv_decode2 = expansive_block(chn3, chn2, chn1)
        self.final_layer = final_block(chn2, chn1, out_channel)

        # Max Pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        # Encoder, or contracting path, formed by 3 contracting blocks followed each one by a Max Pooling layer
        # After each contracting block, a Max Pooling layer halves the size of the image
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.maxpool(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.maxpool(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.maxpool(encode_block3)

        # Decoder, or expansive path, formed by 3 expansive blocks with concatenations, with a final layer at the end
        bottleneck1 = self.bottleneck(encode_pool3)
        decode_block3 = crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return  final_layer

# Encoder network
# Same as the encoder in the U-Net, used for the astrophysical network
class Encoder(nn.Module):

    def __init__(self, in_channel=1, out_channel=chn3):
        super(Encoder, self).__init__()

        #Encoder
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=chn1)
        self.conv_encode2 = contracting_block(chn1, chn2)
        self.conv_encode3 = contracting_block(chn2, chn3)

        # Max Pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        # Encoder, or contracting path, formed by 3 contracting blocks followed each one by a Max Pooling layer
        # After each contracting block, a Max Pooling layer halves the size of the image
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.maxpool(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.maxpool(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.maxpool(encode_block3)
        return  encode_pool3

# Astrophysical network, to perform regression with the astrophysical parameters
# Encoder followed by a secondary convolutional network
class AstroNet(nn.Module):

    def __init__(self,in_channel=1, mid_channel=chn3, out_channel=3):
        super(AstroNet, self).__init__()

        # Encoder from the U-Net
        self.encoder = Encoder(in_channel,mid_channel)

        # Convolutional blocks
        # Now with stride 2 for downsampling, no Max Pooling is used
        self.block1 = miniblock(mid_channel, mid_channel*2, kernel_size=3, stride=2)
        self.block2 = miniblock(mid_channel*2, mid_channel*4, kernel_size=3, stride=2)
        self.block3 = miniblock(mid_channel*4, mid_channel*4, kernel_size=3, stride=2)

        # Input channels for the 1st fully connected layer
        self.n_fc = mid_channel*4*(4)**2

        # Fully connected layers
        self.fc1 = nn.Linear(self.n_fc,100)
        self.fc2 = nn.Linear(100,out_channel)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.20)

    def forward(self, x):

        # Encoder
        x = self.encoder(x)

        # 3 unit layers with stride 2 for downsampling
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Flatten the array and get 3 values for the astrophysical parameters
        x = x.view(-1, self.n_fc)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
