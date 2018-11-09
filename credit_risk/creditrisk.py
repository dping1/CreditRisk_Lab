from __future__ import print_function


import logging
import gzip
import mxnet as mx
import numpy as np
import os
import struct

os.system('pip install pandas')

import pandas as pd


def load_data(path):
    input_file = pd.read_csv(path)
    X = input_file.drop('default.payment.next.month',axis=1)
    y = input_file['default.payment.next.month']
    return y, X


def find_file(root_path, file_name):
    for root, dirs, files in os.walk(root_path):
        if file_name in files:
            return os.path.join(root, file_name)


def build_graph():
    
    
    input_x = mx.sym.Variable('data')
    input_y = mx.sym.Variable('softmax_label')

    fc1  = mx.sym.FullyConnected(data=input_x, num_hidden=250)
    act1 = mx.sym.Activation(data=fc1, act_type="relu") 
    fc2  = mx.sym.FullyConnected(data=act1, num_hidden=250)
    act2 = mx.sym.Activation(data=fc1, act_type="relu") 

    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=2) 

    mlp = mx.sym.SoftmaxOutput(data=fc3, label=input_y, name='softmax')
    return mlp


def train(current_host, channel_input_dirs, hyperparameters, hosts, num_cpus, num_gpus):
    
    batch_size = 100
    learning_rate = hyperparameters.get("learning_rate", 0.1)
    
    training_dir = channel_input_dirs['training']
    
    (train_Y, train_X) = load_data(training_dir + '/train/train.csv')
    (test_Y, test_X) = load_data(training_dir + '/test/test.csv')
    
     
    train_iter = mx.io.NDArrayIter(train_X.values, train_Y.values, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_X.values, test_Y.values, batch_size)
    logging.getLogger().setLevel(logging.DEBUG)
    
    kvstore = 'local' if len(hosts) == 1 else 'dist_sync'
    
    mlp_model = mx.mod.Module(
        symbol=build_graph(),
        context=get_train_context(num_cpus, num_gpus))
    
    mlp_model.fit(train_iter,
                  eval_data=val_iter,
                  kvstore=kvstore,
                  optimizer='adam',
                  optimizer_params={'learning_rate': learning_rate},
                  eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  num_epoch=25)
    return mlp_model


def get_train_context(num_cpus, num_gpus):
    if num_gpus > 0:
        return mx.gpu()
    return mx.cpu()


