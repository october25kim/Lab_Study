import os
import libmr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import gzip
import pickle
from time import time
from tqdm import tqdm

sys.path.append('..')

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.datasets.cifar10 import load_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

dev = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(dev[0], True)
experiment_start = time()

''' Experiment Configurations '''
SEED = 55
SAVEDIR = '/result/'
TARGETS = '0,1,2,3,4,5'
MODEL = 'Openmax'
DATASET = 'Hyundai_Car'

experiment_start = time()

total_classes = list(range(5))
target_classes = total_classes
m = len(target_classes)

BATCH_SIZE = 32
eta = 10
threshold = 0.9

BUFFER_SIZE = 20000
TEST_BATCH_SIZE = 32

BASENAME = '{}-{}-{}'.format(DATASET, MODEL, SEED)

os.makedirs(SAVEDIR, exist_ok=True)
tf.random.set_seed(SEED)

def ConvBlock(x, filters):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters * 4, use_bias=False,
               kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, use_bias=False,
               kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    return x


def TransitionBlock(x, filters, compression=1):
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = Conv2D(filters=int(filters * compression), use_bias=False,
               kernel_size=(1, 1), strides=(1, 1),
               padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x


def DenseBlock(x, layers, growth_rate):
    concat_feature = x
    for l in range(layers):
        x = ConvBlock(concat_feature, growth_rate)
        concat_feature = Concatenate(axis=-1)([concat_feature, x])
    return concat_feature


def define_model(x_shape, use_bias=False, print_summary=False):
    _in = Input(shape=x_shape)
    x = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1),
               padding='same', use_bias=False)(_in)
    x = DenseBlock(x, 16, 12)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 16, 12)
    x = TransitionBlock(x, x.shape[-1], 0.5)
    x = DenseBlock(x, 16, 12)
    x = GlobalAveragePooling2D()(x)
    _out = Dense(units=m, use_bias=False, activation=None)(x)
    model = tf.keras.models.Model(inputs=_in, outputs=_out, name='DenseNet')
    if print_summary:
        model.summary()
    return model

with gzip.open('D:/Openset_signal/data/X', 'rb') as f:
    X = pickle.load(f)
with gzip.open('D:/Openset_signal/data/Y', 'rb') as f:
    Y = pickle.load(f)
Y = np.array(Y)

y_0 = Y[Y == 0][:2000]
y_6413 = Y[Y == 6413]
y_3000 = Y[Y == 3000]
y_66950 = Y[Y == 66950]
y_25050 = Y[Y == 25050]
y_20052 = Y[Y == 20052]

x_0 = X[np.tile(Y == 0, 52*60).reshape(-1,52,60)].reshape(-1,52,60)[:2000]
x_6413 = X[np.tile(Y == 6413, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_3000 = X[np.tile(Y == 3000, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_66950 = X[np.tile(Y == 66950, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_25050 = X[np.tile(Y == 25050, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_20052 = X[np.tile(Y == 20052, 52*60).reshape(-1,52,60)].reshape(-1,52,60)

y = np.append(y_0,y_3000, axis=0)
y = np.append(y,y_66950, axis=0)
y = np.append(y,y_25050, axis=0)
y = np.append(y,y_20052, axis=0)

x = np.append(x_0,x_3000, axis=0)
x = np.append(x,x_66950, axis=0)
x = np.append(x,x_25050, axis=0)
x = np.append(x,x_20052, axis=0)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1)

train_x = np.expand_dims(train_x, axis=3)
test_x = np.expand_dims(test_x, axis=3)
train_x, test_x = train_x[:,10:52,:].astype(np.float16), test_x[:,10:52,:].astype(np.float16)

enc = OneHotEncoder(sparse=False, categories='auto')
train_y_enc = enc.fit_transform(train_y.reshape(-1, 1)).astype(np.float16)
test_y_enc = enc.fit_transform(test_y.reshape(-1, 1)).astype(np.float16)

train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y_enc)).shuffle(BUFFER_SIZE, SEED, True).batch(BATCH_SIZE)
CNN = define_model(train_x.shape[1:], False, False)
network_opt = tf.optimizers.Adam(1E-2)
CNN.summary()
@tf.function
def network_train_step(x, y):
    with tf.GradientTape() as network_tape:
        y_pred = CNN(x, training=True)

        network_loss = tf.keras.losses.categorical_crossentropy(y, y_pred, from_logits=True)
        network_acc = tf.keras.metrics.categorical_accuracy(y, y_pred)

    network_grad = network_tape.gradient(network_loss, CNN.trainable_variables)
    network_opt.apply_gradients(zip(network_grad, CNN.trainable_variables))

    return tf.reduce_mean(network_loss), tf.reduce_mean(network_acc)

def train(dataset, epochs):
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        if epoch == 50:
            network_opt.__setattr__('learning_rate', 1E-3)
        elif epoch == 100:
            network_opt.__setattr__('learning_rate', 1E-4)

        avg_loss = []
        for batch in dataset:
            losses = network_train_step(batch[0], batch[1])
            avg_loss.append(losses)

        pbar.set_description('Categorical CE Loss: {:.4f} | Accuracy: {:.4f} '.format(*np.array(losses)))


def get_model_outputs(dataset, prob=False):
    pred_scores = []
    for x in dataset:
        model_outputs = CNN(x, training=False)
        if prob:
            model_outputs = tf.nn.softmax(model_outputs)
        pred_scores.append(model_outputs.numpy())
    pred_scores = np.concatenate(pred_scores, axis=0)
    return pred_scores

train(train_data, 150)

train_data = tf.data.Dataset.from_tensor_slices(train_x).batch(TEST_BATCH_SIZE)
train_pred_scores = get_model_outputs(train_data, False)
train_pred_simple = np.argmax(train_pred_scores, axis=1)
train_y_a = np.argmax(train_y_enc, axis=1)
accuracy_score(train_y_a, train_pred_simple)

train_correct_actvec = train_pred_scores[np.where(train_y_a == train_pred_simple)[0]]
train_correct_labels = train_y[np.where(train_y_a == train_pred_simple)[0]]
dist_to_means = []
mr_models, class_means = [], []

for c in np.unique(train_y):
    class_act_vec = train_correct_actvec[np.where(train_correct_labels == c)[0], :]
    class_mean = class_act_vec.mean(axis=0)
    dist_to_mean = np.square(class_act_vec - class_mean).sum(axis=1)
    dist_to_mean = np.sort(dist_to_mean).astype(np.float64)
    dist_to_means.append(dist_to_mean)

    mr = libmr.MR()
    mr.fit_high(dist_to_mean[-eta:], eta)

    class_means.append(class_mean)
    mr_models.append(mr)

class_means = np.array(class_means)

def compute_openmax(actvec):
    dist_to_mean = np.square(actvec - class_means).sum(axis=1).astype(np.float64)
    scores = []
    for dist, mr in zip(dist_to_mean, mr_models):
        scores.append(mr.w_score(dist))
    scores = np.array(scores)
    w = 1 - scores
    rev_actvec = np.concatenate([
        w * actvec,
        [((1 - w) * actvec).sum()]])
    return np.exp(rev_actvec) / np.exp(rev_actvec).sum()


def make_prediction(_scores, _T, thresholding=True):
    _scores = np.array([compute_openmax(x) for x in _scores])

    if thresholding:
        uncertain_idx = np.where(np.max(_scores, axis=1) < _T)[0]
        uncertain_vec = np.zeros((len(uncertain_idx), m + 1))
        uncertain_vec[:, -1] = 1

        _scores[uncertain_idx] = uncertain_vec
    _labels = np.argmax(_scores, 1)
    return _labels

thresholding = True
test_data = tf.data.Dataset.from_tensor_slices(test_x).batch(TEST_BATCH_SIZE)
test_pred_scores = get_model_outputs(test_data)
test_pred_labels = make_prediction(test_pred_scores, threshold, thresholding)
test_pred_before = np.argmax(test_pred_scores, axis=1)

## testing on 6413 (Unseen Classes)
unseen_alarm = np.expand_dims(x_6413, axis=3)[:,10:52,:].astype(np.float16)
unseen_alarm_test = tf.data.Dataset.from_tensor_slices(unseen_alarm).batch(TEST_BATCH_SIZE)
test_scores = get_model_outputs(unseen_alarm_test)
test_unseen_alarm_labels = make_prediction(test_scores, threshold, thresholding)
test_before = np.argmax(test_scores, axis=1)

## testing on random noise (Unseen Classes)
images = np.random.uniform(0, 1, (100, 42, 60, 1)).astype(np.float16)
test_batcher = tf.data.Dataset.from_tensor_slices(images).batch(TEST_BATCH_SIZE)
test_scores = get_model_outputs(test_batcher)
test_noise_labels = make_prediction(test_scores, threshold, thresholding)
test_noise_before = np.argmax(test_scores, axis=1)

## Total
test_unseen_labels = np.concatenate([test_unseen_alarm_labels,test_noise_labels])
test_pred = np.concatenate([test_pred_labels, test_unseen_labels])
test_y_a = np.argmax(test_y_enc, axis=1)
test_true = np.concatenate([test_y_a.flatten(), np.ones_like(test_unseen_labels) * m])

test_macro_f1 = f1_score(test_true, test_pred, average='macro')
test_seen_acc = accuracy_score(test_true, test_pred)
confusion_matrix(test_true, test_pred)

## 개별
f1_score(test_y_a, test_pred_before, average='macro')
accuracy_score(test_y_a, test_pred_before)
confusion_matrix(test_y_a, test_pred_before)

test_unseen_f1 = np.array([f1_score(np.ones_like(test_unseen_labels), test_unseen_labels == m),
                           f1_score(np.ones_like(test_unseen_alarm_labels), test_unseen_alarm_labels == m),
                           f1_score(np.ones_like(test_noise_labels), test_noise_labels == m)])

test_unseen_accuracy = np.array([accuracy_score(np.ones_like(test_unseen_labels), test_unseen_labels == m),
                           accuracy_score(np.ones_like(test_unseen_alarm_labels), test_unseen_alarm_labels == m),
                           accuracy_score(np.ones_like(test_noise_labels), test_noise_labels == m)])

confusion_matrix(np.ones_like(test_unseen_labels), test_unseen_labels == m)
confusion_matrix(np.ones_like(test_unseen_alarm_labels), test_unseen_alarm_labels == m)
confusion_matrix(np.ones_like(test_noise_labels), test_noise_labels == m)

print('overall f1: {:.4f}'.format(test_macro_f1))
print('seen acc: {:.4f}'.format(test_seen_acc))
print('unseen f1: {:.4f} / {:.4f} / {:.4f}'.format(*test_unseen_f1))
print('unseen accuracy: {:.4f} / {:.4f} / {:.4f}'.format(*test_unseen_accuracy))