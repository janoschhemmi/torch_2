
import matplotlib as mpl
# mpl.use(‘TkAgg’)

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch import nn
import numpy as np
#import unfoldNd
from inspect import getmembers, isfunction

import tsai
from tsai.all import *
from __future__ import annotations
from tsai.imports import *
from tsai.utils import *
from matplotlib.pyplot import figure
import torch_funs
from matplotlib.collections import LineCollection

import matplotlib.dates as mdates
from datetime import datetime

import fastai
# help(fastai2)
from fastai.vision import *
from fastai.text import *
from fastai.metrics import *
from fastai.learner import *
from fastai.basics import *

import fastcore
from fastcore.all import *
import pandas as pd

## load disturbance time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\02_df_x_8_2.csv', 'r') as f:
    X = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1).reshape(((1498),5,17))
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\02_df_Y_8_2.csv', 'r') as f:
    y = np.genfromtxt(f, delimiter=';',dtype=np.str, skip_header=1)

## split data
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
X.shape, y.shape, splits

# batch_tfms = TSStandardize(by_sample=True)
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

## set up data loader
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[40], batch_tfms=[TSStandardize()], num_workers=0)
# dls.show_batch(sharey=True, nrows = 3, ncols = 3, title = "one batch")

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
learn = Learner(dls, model, metrics=accuracy)
#learn.save('P:/workspace/jan/fire_detection/dl/models/01_test/03_fcn.pth')

#learn.lr_find()
learn.fit_one_cycle(30, lr_max=1e-3)
#learn.save('P:/workspace/jan/fire_detection/dl/models/01_test/03_fcn_stage1.pth')

#learn.recorder.plot_metrics()
learn.save_all(path='P:/workspace/jan/fire_detection/dl/models/01_test/04_fcn_stage2_window_17_smps_300_5_classes.pth', dls_fname='dls', model_fname='model', learner_fname='learner')

##
#
learn = load_learner_all(path='P:/workspace/jan/fire_detection/dl/models/01_test/04_fcn_stage2_window_17_smps_300_5_classes.pth',
                         dls_fname='dls', model_fname='model', learner_fname='learner')
#dls = learn.dls

#learn.dls = dls
#valid_dl = dls.valid
#b = next(iter(valid_dl))
#b

#valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)
#(valid_targets == valid_preds).float().mean()
#dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[40], batch_tfms=[TSStandardize()], num_workers=0)
#learn.dls = dls
#learn.show_results()
#plt.savefig('P:/workspace/jan/fire_detection/dl/models/01_test/03_fcn_stage_2_results_test.png')

## show probabilities
#dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[40], batch_tfms=[TSStandardize()], num_workers=0)
#learn.dls = dls
#learn.show_probas()
#plt.savefig('P:/workspace/jan/fire_detection/dl/models/01_test/03_fcn_stage_2_probas_test.png')

#interp = ClassificationInterpretation.from_learner(learn)



## ## ## ## ## ## ## ## ## ##
## predict on other data

#with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\02_df_x_10_2.csv', 'r') as f:
#    X_valid = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1).reshape((799,5,21))
#with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\02_df_Y_10_2.csv', 'r') as f:
#    y_valid = np.genfromtxt(f, delimiter=';',dtype=np.str, skip_header=1)

#dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[40], batch_tfms=[TSStandardize()], num_workers=0)
#learn.dls = dls
#valid_dl = dls.valid
#test_ds = valid_dl.dataset.add_test(X_valid, y_valid)# In this case I'll use X and y, but this would be your test data
#test_dl = valid_dl.new(test_ds)

#test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
#test_probas, test_targets, test_preds

## built Confusion Matrix
import sklearn
from sklearn import *
from sklearn.metrics import confusion_matrix

#cf_matrix = confusion_matrix(test_targets, test_preds)
#classes = ['Fire','Harvest','Insect','Stable']
#print(cf_matrix)


## ############################
## SPLIT WINDOW APPROACH

## load complete time series data
#with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\03_df_x_50000_ts.csv', 'r') as f:
#    X_ts = np.genfromtxt(f, delimiter=';', dtype=np.float32, skip_header=1)

#with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\03_df_x_50000_ts.csv', 'r') as f:
#    X_ts = pd.read_csv(f, delimiter=';')

## load time series
with open(r'P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\03_df_x_100000_ts_long.csv', 'r') as f:
    X_ts = pd.read_csv(f, delimiter=';')

## load learner
learn = load_learner_all(path='P:/workspace/jan/fire_detection/dl/models/01_test/04_fcn_stage2_window_17_smps_300_5_classes.pth',
                         dls_fname='dls', model_fname='model', learner_fname='learner')

## load references
with open(r'P:\workspace\jan\fire_detection\disturbance_ref\bb_timesync_reference_with_post3_revisited.csv', 'r') as f:
    refs = pd.read_csv(f, delimiter=';')

## get unique ids
ids = X_ts['id'].unique()

## set globals
window_len=17

base_path = r'P:\workspace\jan\fire_detection\dl\plots\01_FCN\window_size_17'
name_base = "_prediction_5_classes_17_window.png"
if os.path.exists(base_path):
    print("path exists")
else:
    os.mkdir(base_path)

## loop over ids
for id in ids[1:100]:
    print(id)
    id = 3298

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

    sample = sample.dropna(axis = 1, how = 'any').astype('int32')
    # length of time series
    #n_steps = len(sample.columns)

    ## sliding window
    ## into np
    sample_np = sample.to_numpy()
    sample_windowed = np.lib.stride_tricks.sliding_window_view(sample_np, (len(index),window_len ))
    sample_windowed = sample_windowed[0,:,:,:]

    ## predict for windows
    probs, test_targets, test_preds = learn.get_X_preds(sample_windowed)
    probs, test_targets, test_preds

    # plot it

    ## time series
    file_name = str(id) + name_base
    ## init plot
    #figure(figsize=(30, 5), dpi=100)
    fig, (ax_1, ax_2) = plt.subplots(2, figsize = (30,8),sharex=True)
    fig.suptitle('prediction and ts of plot id: ' + str(id))

    ## set dates
    formatter = mdates.DateFormatter("%Y")  ### formatter of the date
    locator = mdates.YearLocator()  ### where to put the labels

    ax_2.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_2.xaxis.set_major_locator(locator)  ## calling the locator for the x-axis
    #dates =  datetime.strptime(list(sample.columns), "%Y-%m-%d")
    dates =  [datetime.strptime(date, "%Y-%m-%d") for date in list(sample.columns)]
    #ax1.plot(2, 1, 1)
    ax_2.plot(dates, sample.T)
    ax_2.legend(list(sample.index), fontsize = 20,bbox_to_anchor=(1.01, 1), loc="upper left")
    #ax2.tight_layout()

    ## predictions
    ## get dates
    dates_prediction = dates[int((window_len - 1) / 2):(len(dates)-int((window_len-1) / 2))]

    ## get predictions
    probs_np = probs.numpy()

    ax_1.xaxis.set_major_formatter(formatter)  ## calling the formatter for the x-axis
    ax_1.xaxis.set_major_locator(locator)  ##

    h = ax_1.plot(dates_prediction, probs_np)
    h[3].set_alpha(0.4)
    h[4].set_alpha(0.7)
    h[3].set_linestyle('dashdot')
    h[4].set_linestyle(':')

    ax_1.legend(['fire','harvest','insect','stable','growth'], fontsize=20,bbox_to_anchor=(1.01, 1), loc="upper left")

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


    ## set path name
    tile_path = os.path.join(base_path, tile_id)
    if os.path.exists(tile_path):
        print("path exists")
    else:
        os.mkdir(tile_path)

    fig.tight_layout()
    #figure(figsize=(30, 5), dpi=100)
    print("plot..")
    plt.savefig(os.path.join(tile_path, file_name), dpi = 100)
    plt.clf()

## splits = [x[::;i:i+window_size] for i in range(0,x.size(0)-window_size+1,stride)]


## model on splits
## preds = [model.predict(i) for i in img_arr]

## safe model outputs per sampe