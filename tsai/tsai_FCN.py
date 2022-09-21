
import matplotlib as mpl
# mpl.use(‘TkAgg’)

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from inspect import getmembers, isfunction
import tsai
from tsai.all import *
from __future__ import annotations
from tsai.imports import *
from tsai.utils import *
from tsai.basics import *
from matplotlib.pyplot import figure
#import torch_funs
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from datetime import datetime
import fastai
# help(fastai2)
from fastai.vision import *
from fastai.text import *
from fastai.metrics import *
from fastai.learner import *
from fastai.data import  *
import fastcore
from fastcore.all import *
import pandas as pd
import sklearn
from sklearn import *
from sklearn.metrics import confusion_matrix

plt.switch_backend('agg')

##
## load disturbance time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_x_10_250smps.csv', 'r') as f:
    X = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1).reshape(((1494),8,21))
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_y_10_250smps.csv', 'r') as f:
    y = np.genfromtxt(f, delimiter=';',dtype=np.int32, skip_header=1)

## split data
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
X.shape, y.shape, splits
y


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

## set up data loader
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64], batch_tfms=[TSStandardize()], num_workers=0)


## built learner
class FCNN(Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
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
model = FCNN(dls.vars, dls.c)
model = XceptionModule(dls.vars, dls.c)
#learn = Learner(dls, model, metrics=accuracy)

model = build_ts_model( XceptionTime, dls=dls)
learn = Learner(dls, model, metrics=accuracy)

#learn.save('P:/workspace/jan/fire_detection/dl/models_store/01_test/03_fcn.pth')

#learn.lr_find()
learn.fit_one_cycle(50, lr_max=1e-3, wd = 0.1)

model_path = 'P:/workspace/jan/fire_detection/dl/models_store/04_FCNN/07_XceptionTime_window_21_smps_250_6_classes_3.pth'
learn.save_all(path=model_path, dls_fname='dls', model_fname='model', learner_fname='learner')



learn = load_learner_all(path=model_path,
                         dls_fname='dls', model_fname='model', learner_fname='learner')
learn.dls = dls
valid_dl = dls.valid

## load time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\04_df_x_800000_ts_long.csv', 'r') as f:
    X_ts = pd.read_csv(f, delimiter=';',
                       dtype={
                           'x' :   'int64',
                           'y' :    'int64',
                           'id':    'int64',
                           'date' :  'object',
                           'sensor':  'object',
                           'value':    'float32',
                           'tile':    'object',
                            'index' :  'object',
                            'change_process': 'object',
                             'diff' :  'float64',
                              'instance': 'float64',
                            'time_sequence' : 'int64'
                       }
                       )

X_ts.dtypes
## load learner
learn = load_learner_all(path=model_path,
                         dls_fname='dls', model_fname='model', learner_fname='learner')

print(learn)
learn.dls = dls
valid_dl = dls.valid

## load references
with open(r'P:\workspace\jan\fire_detection\disturbance_ref\bb_timesync_reference_with_wind.csv', 'r') as f:
    refs = pd.read_csv(f, delimiter=';')

refs_to_plot = ['Harvest','Fire','Insect','Wind']
refs = refs[refs['change_process'].isin(refs_to_plot)]
refs['change_process'].unique()
## get unique ids
ids = X_ts['id'].unique()

## set globals
window_len=21
model_name = 'XceptionTime'

base_path = r'P:\workspace\jan\fire_detection\dl\plots\03_FCN\window_size_21_6_classes'
name_base = "_prediction_6_classes_21_window_2_XCeptionTime.png"
if os.path.exists(base_path):
    print("path exists")
else:
    os.mkdir(base_path)

## loop over ids
for id in ids[1:250]:
    print(id)
    #id = 1059

    ## subset df for each id
    #   subset = X_ts[X_ts[:,2] == id,:]
    sample = X_ts[X_ts['id']==id]
    # sample = X_ts[X_ts['id'] == 947]

    ## store tile id of sample
    tile_id = sample['tile'].values[0]

    ## into long format
    sample = sample.pivot(index='index', columns='date', values='value')

    index = list(sample.index)
    dates = list(sample.columns)

    sample = sample.dropna(axis = 1, how = 'any').astype('float32')
    # length of time series
    #n_steps = len(sample.columns)

    ## into np
    sample_np = sample.to_numpy(dtype = 'float32')

    #sample_np.dtype
    sample_windowed = np.lib.stride_tricks.sliding_window_view(sample_np, (len(index),window_len ))
    sample_windowed = sample_windowed[0,:,:,:]
    torch.from_numpy(sample_windowed)
    #sample_windowed.dtype
    ## predict for windows
    sample_windowed
    probs, test_targets, test_preds = learn.get_X_preds((sample_windowed))
    probs, test_targets, test_preds

    # plot it

    ## time series
    file_name = str(id) + name_base
    ## init plot
    #figure(figsize=(30, 5), dpi=100)
    fig, (ax_1, ax_2) = plt.subplots(2, figsize = (30,10),sharex=True)
    fig.suptitle(model_name + ' prediction and ts of plot id: ' + str(id), fontsize = 20)

    ## set dates
    formatter = mdates.DateFormatter("%Y")  ### formatter of the date
    locator = mdates.YearLocator()  ### where to put the labels

    ax_2.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_2.xaxis.set_major_locator(locator)  ## calling the locator for the x-axis
    #dates =  datetime.strptime(list(sample.columns), "%Y-%m-%d")
    dates =  [datetime.strptime(date, "%Y-%m-%d") for date in list(sample.columns)]
    #ax1.plot(2, 1, 1)
    ax_2.plot(dates, sample.T)
    ax_2.legend(list(sample.index), fontsize = 22,bbox_to_anchor=(1.01, 1), loc="upper left")
    #ax2.tight_layout()

    ## predictions
    ## get dates
    dates_prediction = dates[int((window_len - 1) / 2):(len(dates)-int((window_len-1) / 2))]

    ## get predictions
    probs_np = probs.numpy()

    ax_1.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_1.xaxis.set_major_locator(locator)  ##

    h = ax_1.plot(dates_prediction, probs_np)
    h[4].set_alpha(0.4)
    h[5].set_alpha(0.7)
    h[4].set_linestyle('dashdot')
    h[5].set_linestyle(':')

    h[0].set_color("red")
    h[1].set_color("brown")
    h[2].set_color("goldenrod")
    h[3].set_color("blue")
    h[4].set_color("black")
    h[5].set_color("green")

    ax_1.legend(['fire','harvest','insect','wind','stable','growth'], fontsize=20,bbox_to_anchor=(1.01, 1), loc="upper left")

    ## get refs
    ref_id = refs[(refs["plotid"] == id) & (refs["disturbance"]==1) ]
    dates_refs = [datetime.strptime(ref_d, "%Y-%m-%d") for ref_d in list(ref_id["change_date"])]
    change_label = list(ref_id["change_process"])

    ## add refs
    y_min, y_max = ax_1.get_ylim()
    ax_1.vlines(x=dates_refs, ymin=y_min, ymax=y_max, color='k', ls='--')
    y_min, y_max = ax_2.get_ylim()
    ax_2.vlines(x=dates_refs, ymin=y_min, ymax=y_max, color='k', ls='--')

    [ax_2.text(dates_refs[i],y= (y_max - (y_max / 12 ) ) , s = change_label[i], fontfamily = 'monospace', fontsize = 'xx-large',fontstyle = 'italic',
               verticalalignment = 'top', fontweight = 'roman', ha = 'right')
     for i in range(len(dates_refs))]


    ax_1.tick_params(axis='both', which='major', labelsize=16)
    ax_1.tick_params(axis='both', which='minor', labelsize=16)
    ax_2.tick_params(axis='both', which='major', labelsize=16)
    ax_2.tick_params(axis='both', which='minor', labelsize=16)
    ## set path name
    tile_path = os.path.join(base_path, tile_id)
    if os.path.exists(tile_path):
        print("path exists")
    else:
        os.mkdir(tile_path)

    fig.tight_layout()
    #figure(figsize=(30, 5), dpi=100)
    print("plot..")
    plt.savefig(os.path.join(tile_path, file_name), dpi = 120)
    plt.clf()






#################################################################################################################################################
############################################################################################################################


## load disturbance time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_x_10_250smps.csv', 'r') as f:
    X = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1).reshape(((1494), 8, 21))
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_y_10_250smps.csv', 'r') as f:
    y = np.genfromtxt(f, delimiter=';', dtype=np.int32, skip_header=1)

## split data
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
X.shape, y.shape, splits
y

tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

## set up data loader
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64], batch_tfms=[TSStandardize()], num_workers=0)

## built learner
model = build_ts_model(LSTM_FCNPlus, dls=dls)
learn = Learner(dls, model, metrics=accuracy)

# learn.save('P:/workspace/jan/fire_detection/dl/models_store/01_test/03_fcn.pth')

# learn.lr_find()
learn.fit_one_cycle(50, lr_max=1e-4, wd=0.1)

model_path = 'P:/workspace/jan/fire_detection/dl/models_store/04_FCNN/08_LSTM_FCNPlus_window_21_smps_250_6_classes_2.pth'
learn.save_all(path=model_path, dls_fname='dls', model_fname='model', learner_fname='learner')

#learn = load_learner_all(path=model_path,
#                         dls_fname='dls', model_fname='model', learner_fname='learner')
learn.dls = dls
valid_dl = dls.valid

## load time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\04_df_x_800000_ts_long.csv', 'r') as f:
    X_ts = pd.read_csv(f, delimiter=';',
                       dtype={
                           'x': 'int64',
                           'y': 'int64',
                           'id': 'int64',
                           'date': 'object',
                           'sensor': 'object',
                           'value': 'float32',
                           'tile': 'object',
                           'index': 'object',
                           'change_process': 'object',
                           'diff': 'float64',
                           'instance': 'float64',
                           'time_sequence': 'int64'
                       }
                       )

X_ts.dtypes
## load learner
#learn = load_learner_all(path=model_path,
#                         dls_fname='dls', model_fname='model', learner_fname='learner')

print(learn)
learn.dls = dls
valid_dl = dls.valid

## load references
with open(r'P:\workspace\jan\fire_detection\disturbance_ref\bb_timesync_reference_with_wind.csv', 'r') as f:
    refs = pd.read_csv(f, delimiter=';')

refs_to_plot = ['Harvest', 'Fire', 'Insect', 'Wind']
refs = refs[refs['change_process'].isin(refs_to_plot)]
refs['change_process'].unique()
## get unique ids
ids = X_ts['id'].unique()

## set globals
window_len = 21
model_name = 'LSTM_FCNPlus'

base_path = r'P:\workspace\jan\fire_detection\dl\plots\03_FCN\window_size_21_6_classes'
name_base = "_prediction_6_classes_21_window_2_LSTM_FCNPlus.png"
if os.path.exists(base_path):
    print("path exists")
else:
    os.mkdir(base_path)

## loop over ids
for id in ids[1:250]:
    print(id)
    # id = 1059

    ## subset df for each id
    #   subset = X_ts[X_ts[:,2] == id,:]
    sample = X_ts[X_ts['id'] == id]
    # sample = X_ts[X_ts['id'] == 947]

    ## store tile id of sample
    tile_id = sample['tile'].values[0]

    ## into long format
    sample = sample.pivot(index='index', columns='date', values='value')

    index = list(sample.index)
    dates = list(sample.columns)

    sample = sample.dropna(axis=1, how='any').astype('float32')
    # length of time series
    # n_steps = len(sample.columns)

    ## into np
    sample_np = sample.to_numpy(dtype='float32')

    # sample_np.dtype
    sample_windowed = np.lib.stride_tricks.sliding_window_view(sample_np, (len(index), window_len))
    sample_windowed = sample_windowed[0, :, :, :]
    torch.from_numpy(sample_windowed)
    # sample_windowed.dtype
    ## predict for windows
    sample_windowed
    probs, test_targets, test_preds = learn.get_X_preds((sample_windowed))
    probs, test_targets, test_preds

    # plot it

    ## time series
    file_name = str(id) + name_base
    ## init plot
    # figure(figsize=(30, 5), dpi=100)
    fig, (ax_1, ax_2) = plt.subplots(2, figsize=(30, 10), sharex=True)
    fig.suptitle(model_name + ' prediction and ts of plot id: ' + str(id), fontsize=20)

    ## set dates
    formatter = mdates.DateFormatter("%Y")  ### formatter of the date
    locator = mdates.YearLocator()  ### where to put the labels

    ax_2.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_2.xaxis.set_major_locator(locator)  ## calling the locator for the x-axis
    # dates =  datetime.strptime(list(sample.columns), "%Y-%m-%d")
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in list(sample.columns)]
    # ax1.plot(2, 1, 1)
    ax_2.plot(dates, sample.T)
    ax_2.legend(list(sample.index), fontsize=22, bbox_to_anchor=(1.01, 1), loc="upper left")
    # ax2.tight_layout()

    ## predictions
    ## get dates
    dates_prediction = dates[int((window_len - 1) / 2):(len(dates) - int((window_len - 1) / 2))]

    ## get predictions
    probs_np = probs.numpy()

    ax_1.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_1.xaxis.set_major_locator(locator)  ##

    h = ax_1.plot(dates_prediction, probs_np)
    h[4].set_alpha(0.4)
    h[5].set_alpha(0.7)
    h[4].set_linestyle('dashdot')
    h[5].set_linestyle(':')

    h[0].set_color("red")
    h[1].set_color("brown")
    h[2].set_color("goldenrod")
    h[3].set_color("blue")
    h[4].set_color("black")
    h[5].set_color("green")

    ax_1.legend(['fire', 'harvest', 'insect', 'wind', 'stable', 'growth'], fontsize=20, bbox_to_anchor=(1.01, 1),
                loc="upper left")

    ## get refs
    ref_id = refs[(refs["plotid"] == id) & (refs["disturbance"] == 1)]
    dates_refs = [datetime.strptime(ref_d, "%Y-%m-%d") for ref_d in list(ref_id["change_date"])]
    change_label = list(ref_id["change_process"])

    ## add refs
    y_min, y_max = ax_1.get_ylim()
    ax_1.vlines(x=dates_refs, ymin=y_min, ymax=y_max, color='k', ls='--')
    y_min, y_max = ax_2.get_ylim()
    ax_2.vlines(x=dates_refs, ymin=y_min, ymax=y_max, color='k', ls='--')

    [ax_2.text(dates_refs[i], y=(y_max - (y_max / 12)), s=change_label[i], fontfamily='monospace', fontsize='xx-large',
               fontstyle='italic',
               verticalalignment='top', fontweight='roman', ha='right')
     for i in range(len(dates_refs))]

    ax_1.tick_params(axis='both', which='major', labelsize=16)
    ax_1.tick_params(axis='both', which='minor', labelsize=16)
    ax_2.tick_params(axis='both', which='major', labelsize=16)
    ax_2.tick_params(axis='both', which='minor', labelsize=16)
    ## set path name
    tile_path = os.path.join(base_path, tile_id)
    if os.path.exists(tile_path):
        print("path exists")
    else:
        os.mkdir(tile_path)

    fig.tight_layout()
    # figure(figsize=(30, 5), dpi=100)
    print("plot..")
    plt.savefig(os.path.join(tile_path, file_name), dpi=120)
    plt.clf()





