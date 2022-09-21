
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## read data
X_train = pd.read_csv(r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_x_8_250smps.csv", sep = ';')
y_train = pd.read_csv(r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_y_8_250smps.csv", sep = ';')
print(X_train.shape, y_train.shape)

X_train
## data labeler
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train.disturbance)
print(encoded_labels[:1000])
print("Label Classes: ", label_encoder.classes_)

# Adding to y_train df
y_train["label"] = encoded_labels
# reshape
X_train = (X_train.set_index(['id', 'index','instances_rep'])
   .rename_axis(['step'], axis=1)
   .stack()
   .unstack('index')
   .reset_index())

FEATURE_COLUMNS = X_train.columns.tolist()[3:]
FEATURE_COLUMNS

## group per id for individual sequence
sequences = []

for instances, group in X_train.groupby("instances_rep"):
    print(id)
    sequence_features = group[FEATURE_COLUMNS]
    label = y_train[y_train.instance_rep == instances].label
    print(label)
    sequences.append((sequence_features, label))
sequences[0][1]

## split train test
train_sequences, test_sequences = train_test_split(sequences, test_size=0.2)
print("Number of Training Sequences: ", len(train_sequences))
print("Number of Testing Sequences: ", len(test_sequences))

##
t_1, t_2 = sequences[10]
label=torch.tensor(t_2).long()

one_sequence = sequences[0]
one_sequence = one_sequence[0][["NBR","NDV","BLU","GRN"]]

one_sequence = torch.Tensor(one_sequence.to_numpy())
one_sequence_tensor = torch.transpose(one_sequence, 0 , 1)
one_sequence_tensor = one_sequence_tensor[None,:,: ]

## defin conv layer in band dimension
c = nn.Conv1d(in_channels = 4, out_channels = 8, stride = 1, kernel_size= 1)
one_sequence_tensor_c1 = c(one_sequence_tensor)
one_sequence_tensor_c1.shape

## 1 conv time layer
c2_time = nn.Conv1d(8, 128, stride=1, kernel_size=5, padding=0)
c2_time.weight.shape
one_sequence_tensor_c2 = c2_time(one_sequence_tensor_c1)
one_sequence_tensor_c2.shape

one_sequence_tensor_c2[0,3,:]

## 1 relu
relu = nn.ReLU()
one_sequence_tensor_c2 = relu(one_sequence_tensor_c2)
bn = nn.BatchNorm1d(128)
one_sequence_tensor_c2 = bn(one_sequence_tensor_c2)

### 2 conv time layer
c2_time = nn.Conv1d(128, 256, stride=1, kernel_size=7, padding=3)
one_sequence_tensor_c2 = c2_time(one_sequence_tensor_c2)

### 3 conv time layer
c3_time = nn.Conv1d(256, 128, stride=1, kernel_size=3, padding=1)
one_sequence_tensor_c3 = c3_time(one_sequence_tensor_c2)

## 4 max pool
one_sequence_tensor_c3.shape
mp = nn.MaxPool2d( (3,2), stride = 1)
one_sequence_tensor_c4 = mp(one_sequence_tensor_c3)
one_sequence_tensor_c4.shape
## flatten
fl = nn.Flatten()
one_sequence_tensor_c5 = fl(one_sequence_tensor_c4)

## linear
c_4_linear = nn.Linear( 16, 126)
one_sequence_tensor_c5 = c_4_linear(one_sequence_tensor_c4)

## linear
c_5_linear = nn.Linear( 2016, 1000)
one_sequence_tensor_c6 = c_5_linear(one_sequence_tensor_c5)

c_6_linear = nn.Linear( 1000, 6)
one_sequence_tensor_c7 = c_6_linear(one_sequence_tensor_c6)


## flatten
fl = nn.Flatten()
one_sequence_tensor_c7 = fl(one_sequence_tensor_c6)

## softmax
sm = nn.Softmax(dim =1)
out = sm(one_sequence_tensor_c7)
