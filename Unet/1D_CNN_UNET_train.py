import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset, DataLoader
import torchvision
import pytorch_lightning as pl

#import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from matplotlib.ticker import MaxNLocator

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from multiprocessing import cpu_count

## load model
import Unet
from Unet import UNET
from Unet import

 ## SETTINGS
# %matplotlib inline
%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
tqdm.pandas()

##
X_train = pd.read_csv("P:/workspace/jan/fire_detection/dl/prepocessed_ref_tables/04_unet_df_x_200smps.csv", sep = ';')
y_train = pd.read_csv("P:/workspace/jan/fire_detection/dl/prepocessed_ref_tables/04_unet_df_y_200smps.csv", sep = ';')
print(X_train.shape, y_train.shape)

# Changing from String Labels to Interger Labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train.change_process)
print(encoded_labels[:1000])
print("Label Classes: ", label_encoder.classes_)

## set order of classes ## if time

# Adding to y_train df
y_train["label"] = encoded_labels

X_train = (X_train.set_index(['id', 'index'])
   .rename_axis(['step'], axis=1)
   .stack()
   .unstack('index')
   .reset_index())

FEATURE_COLUMNS = X_train.columns.tolist()[2:]
FEATURE_COLUMNS

## group per id for individual sequence
sequences = []
for id, group in X_train.groupby("id"):
    print(id)
    sequence_features = group[FEATURE_COLUMNS]
    label = y_train[y_train.id == id].label
    sequences.append((sequence_features, label))
sequences[0]

## split in training and test sequences
train_sequences, valid_sequences = train_test_split(sequences, test_size=0.1)
train_sequences, test_sequences = train_test_split(train_sequences, test_size=0.2)
print("Number of Training Sequences: ", len(train_sequences))
print("Number of Testing Sequences: ", len(test_sequences))
print("Number of Valid Sequences: ", len(valid_sequences))



## model Parameters
N_EPOCHS = 10
BATCH_SIZE = 32

## PYtorch Lightening dataloader
data_module = TSDataModule(train_sequences, test_sequences, BATCH_SIZE)

#Build model, initial weight and optimizer
#model = UNET_1D(1,128,7,3) #(input_dim, hidden_layer, kernel_size, depth)
# UNET = UNET_Predictor(8,64,4,2,6)
UNET = UNET_Predictor(8, 6, 240)

trainer = pl.Trainer(max_epochs=2)
trainer.fit(UNET, data_module)

# label_test
la = sequences[0][1]
la

## TEST
tt = sequences[0]
tt = tt[0][["NBR","NDV"]]
tt = torch.Tensor(tt.to_numpy())
tt = torch.transpose(tt, 0 , 1)
tt = tt[None,:,: ]
c = nn.Conv1d(in_channels = 2, out_channels = 6, stride = 1, kernel_size=(1))
zz = c(tt)
tt = tt[None,None,:,: ]
c = nn.Conv2d(in_channels = 2, out_channels = 6, stride = 1, kernel_size=1)

## expect 1 * 5

m = nn.MaxPool1d( kernel_size = 2, stride=2)
uu = m(zz)