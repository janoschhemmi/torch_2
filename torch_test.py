

import torch
from torch import nn

import numpy as np
from tsai.all import *
import torch
from torch import nn

## load tree species time series
with open('gdrive/MyDrive/Colab Notebooks/tsai/disturbances/data/02_df_x_10.csv', 'r') as f:
    X = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1).reshape((799,5,21))

with open('gdrive/MyDrive/Colab Notebooks/tsai/disturbances/data/02_df_y_10.csv', 'r') as f:
    y = np.genfromtxt(f, delimiter=';',dtype=np.str, skip_header=1)


X.shape, y.shape

splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
splits


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)


dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[50], batch_tfms=[TSStandardize()], num_workers=0)

dls.show_batch(sharey=True,  show_title= True, ncols = 3, nrows= 5)


class FCN(Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0], dilation = 2)
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        return self.fc(x)

model_test = FCN(dls.vars, dls.c,layers=[128, 256, 128], kss=[7, 5, 3])
learn = Learner(dls, model_test, metrics=accuracy)
#learn.load('stage0')
learn.lr_find()

learn.fit_one_cycle(25, lr_max=1e-3)
#learn.save('stage1_2')



unfold = nn.Unfold(kernel_size=(2, 3))
unfold
input = torch.randn( 10, 5, 21)

output = unfold(input)
output


#>>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
#>>> # 4 blocks (2x3 kernels) in total in the 3x4 input
#>>> output.size()
#torch.Size([2, 30, 4])

#>>> # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
#>>> inp = torch.randn(1, 3, 10, 12)
#>>> w = torch.randn(2, 3, 4, 5)
#>>> inp_unf = torch.nn.functional.unfold(inp, (4, 5))
#>>> out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
#>>> out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
#>>> # or equivalently (and avoiding a copy),
#>>> # out = out_unf.view(1, 2, 7, 8)
#>>> (torch.nn.functional.conv2d(inp, w) - out).abs().max()
#tensor(1.9073e-06)