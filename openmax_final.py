import os
import libmr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import gzip
import pickle
import itertools
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# aaa
sys.path.append('..')

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

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

total_classes = list(range(4))
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

with gzip.open('D:/Openset_signal/data/108_세타_m37_2020-04-01_2020-05-31_x', 'rb') as f:
    X = pickle.load(f)
with gzip.open('D:/Openset_signal/data/108_세타_m37_2020-04-01_2020-05-31_y', 'rb') as f:
    Y = pickle.load(f)
Y = np.array(Y)

y_0 = Y[Y == 5]
y_6413 = Y[Y == 6413]
y_3000 = Y[Y == 3000]
y_66950 = Y[Y == 66950]
y_25050 = Y[Y == 25050]
y_20052 = Y[Y == 20052]

x_0 = X[np.tile(Y == 5, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_6413 = X[np.tile(Y == 6413, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_3000 = X[np.tile(Y == 3000, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_66950 = X[np.tile(Y == 66950, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_25050 = X[np.tile(Y == 25050, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_20052 = X[np.tile(Y == 20052, 52*60).reshape(-1,52,60)].reshape(-1,52,60)

plt.plot(x_0[100][26,:], marker='.', label = 'normal')
plt.plot(x_6413[9][26,:], marker='.', c ='r', label = '6413')
plt.plot(x_3000[154][26,:], marker='.', label = '3000')
plt.plot(x_66950[89][26,:], marker='.', label = '66950')
plt.plot(x_25050[100][26,:], marker='.', label = '25050')
plt.plot(x_20052[35][26,:], marker='.', label = '20052')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_0[100][26,:], marker='.', label = 'normal')
plt.plot(x_6413[9][26,:], marker='.',c ='r', label = '6413')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_6413[9][26,:], marker='.', c ='r', label = '6413')
plt.plot(x_20052[35][26,:], marker='.', label = '20052')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_6413[9][26,:], marker='.', c ='r', label = '6413')
plt.plot(x_66950[89][26,:], marker='.', label = '66950')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_6413[9][26,:], marker='.', c ='r', label = '6413')
plt.plot(x_25050[100][26,:], marker='.', label = '25050')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_0[100][26,:], marker='.', label = 'normal')
plt.plot(x_25050[100][26,:], marker='.', label = '25050')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_6413[9][26,:], marker='.', c ='r', label = '6413')
plt.plot(x_3000[154][26,:], marker='.', label = '3000')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.plot(x_0[100][26,:], marker='.', label = 'normal')
plt.plot(x_6413[9][26,:], marker='.', label = '6413')
plt.plot(x_3000[154][26,:], marker='.', label = '3000')
# plt.plot(x_66950[100][10,:], marker='.', label = '66950')
# plt.plot(x_25050[100][10,:], marker='.', label = '25050')
# plt.plot(x_20052[100][10,:], marker='.', label = '20052')
plt.legend(loc='upper right')
plt.grid()
plt.show()

y = np.append(y_0,y_6413, axis=0)
y = np.append(y,y_66950, axis=0)
y = np.append(y,y_3000, axis=0)

x = np.append(x_0,x_6413, axis=0)
x = np.append(x,x_66950, axis=0)
x = np.append(x,x_3000, axis=0)

X = x[:,10:52,:]
X_test = np.append(x_20052, x_25050, axis=0)[:,10:52,:]
y_test = np.append(y_20052, y_25050)

X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size = 0.20, random_state = 2020, stratify = y)

X_train = X_train.transpose(0,2,1)
X_train = X_train.reshape(-1,42)

X_val = X_val.transpose(0,2,1)
X_val = X_val.reshape(-1,42)

X_test = X_test.transpose(0,2,1)
X_test = X_test.reshape(-1,42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_train = X_train_scaled.reshape(-1,60,42)

X_val_scaled = scaler.transform(X_val)
X_val = X_val_scaled.reshape(-1,60,42)

X_test_scaled = scaler.transform(X_test)
X_test = X_test_scaled.reshape(-1,60,42)

X_train = X_train.transpose(0,2,1)
X_val = X_val.transpose(0,2,1)
X_test = X_test.transpose(0,2,1)

train_x = np.expand_dims(X_train, axis=3).astype(np.float16)
test_x = np.expand_dims(X_val, axis=3).astype(np.float16)
unseen_x = np.expand_dims(X_test, axis=3).astype(np.float16)

enc = OneHotEncoder(sparse=False, categories='auto')
train_y_enc = enc.fit_transform(y_train.reshape(-1, 1)).astype(np.float16)
test_y_enc = enc.fit_transform(y_val.reshape(-1, 1)).astype(np.float16)

train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y_enc)).shuffle(BUFFER_SIZE, SEED, True).batch(BATCH_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y_enc)).batch(BATCH_SIZE)

CNN = define_model(train_x.shape[1:], False, False)
network_opt = tf.optimizers.Adam(1E-2)

@tf.function
def network_train_step(x, y):
    with tf.GradientTape() as network_tape:
        y_pred = CNN(x, training=True)
        network_loss = tf.keras.losses.categorical_crossentropy(y, y_pred, from_logits=True)
        network_acc = tf.keras.metrics.categorical_accuracy(y, y_pred)
    network_grad = network_tape.gradient(network_loss, CNN.trainable_variables)
    network_opt.apply_gradients(zip(network_grad, CNN.trainable_variables))
    return tf.reduce_mean(network_loss), tf.reduce_mean(network_acc)

def network_valid_step(x, y):
    y_pred = CNN(x, training=True)
    network_loss = tf.keras.losses.categorical_crossentropy(y, y_pred, from_logits=True)
    network_acc = tf.keras.metrics.categorical_accuracy(y, y_pred)
    return tf.reduce_mean(network_loss), tf.reduce_mean(network_acc)

def train(dataset, valid_dataset, epochs):
    pbar = tqdm(range(epochs))
    train_loss = []
    val_loss = []
    for epoch in pbar:
        if epoch == 50:
            network_opt.__setattr__('learning_rate', 1E-3)
        elif epoch == 100:
            network_opt.__setattr__('learning_rate', 1E-4)
        for batch in dataset:
            losses = network_train_step(batch[0], batch[1])
        for batch in valid_dataset:
            val_losses = network_valid_step(batch[0], batch[1])
        pbar.set_description('CE Loss: {:.4f} | Accuracy: {:.4f} | Val CE Loss: {:.4f} | Val Accuracy: {:.4f} '.format(*np.array(losses), *np.array(val_losses)))
        train_loss.append(losses)
        val_loss.append(val_losses)
    return train_loss, val_loss

def get_model_outputs(dataset, prob=False):
    pred_scores = []
    for x in dataset:
        model_outputs = CNN(x, training=False)
        if prob:
            model_outputs = tf.nn.softmax(model_outputs)
        pred_scores.append(model_outputs.numpy())
    pred_scores = np.concatenate(pred_scores, axis=0)
    return pred_scores

train_loss, val_loss = train(train_data, test_data, 100)
CNN.save_weights("D:/Openset_signal/openmax/model/DenseNet_100epochs_big3.h5")
# plotting loss
x_len = np.arange(len(train_loss))
plt.plot(x_len, train_loss, marker='.')
plt.plot(x_len, val_loss, marker='.')
plt.legend(loc='upper right',labels=("Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"))
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(9, 9))
    plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=16, labelpad=20)
    plt.xlabel('Predicted label', fontsize=16, labelpad=20)

    plt.tight_layout()
    plt.grid(False)
    plt.show()

# Train accuracy
train_data = tf.data.Dataset.from_tensor_slices(train_x).batch(TEST_BATCH_SIZE)
train_pred_scores = get_model_outputs(train_data, False)
train_pred_simple = np.argmax(train_pred_scores, axis=1)
train_y_a = np.argmax(train_y_enc, axis=1)
accuracy_score(train_y_a, train_pred_simple)
cm = confusion_matrix(train_y_a, train_pred_simple)
class_list = np.array(['Normal', '3000', '6413', '66950'])
plot_confusion_matrix(cm, classes = class_list, normalize=False, title = 'confusion matrix')

# OpenMax
train_correct_actvec = train_pred_scores[np.where(train_y_a == train_pred_simple)[0]]
train_correct_labels = y_train[np.where(train_y_a == train_pred_simple)[0]]
dist_to_means = []
mr_models, class_means = [], []

eta = 5
for c in np.unique(y_train):
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

def make_prediction2(_scores, _T, thresholding=True):
    _scores = tf.nn.softmax(_scores)
    _scores = _scores.numpy()
    _scores = np.concatenate([_scores, np.expand_dims(np.zeros(len(_scores)), axis=1)], axis=1)

    if thresholding:
        uncertain_idx = np.where(np.max(_scores, axis=1) < _T)[0]
        uncertain_vec = np.zeros((len(uncertain_idx), m + 1))
        uncertain_vec[:, -1] = 1
        _scores[uncertain_idx] = uncertain_vec
    _labels = np.argmax(_scores, 1)
    return _labels

thresholding = True
threshold = 0.99999
print('Threshold = {}'.format(threshold))
test_data = tf.data.Dataset.from_tensor_slices(test_x).batch(TEST_BATCH_SIZE)
test_pred_scores = get_model_outputs(test_data)
test_pred_labels = make_prediction(test_pred_scores, threshold, thresholding)
test_pred_simple = np.argmax(test_pred_scores, axis=1)

## testing on 3000 (Unseen Classes)
alarm_20052_test = tf.data.Dataset.from_tensor_slices(unseen_x[:359]).batch(TEST_BATCH_SIZE)
scores_20052 = get_model_outputs(alarm_20052_test)
alarm_20052_labels = make_prediction(scores_20052, threshold, thresholding)

alarm_25050_test = tf.data.Dataset.from_tensor_slices(unseen_x[359:]).batch(TEST_BATCH_SIZE)
scores_25050 = get_model_outputs(alarm_25050_test)
alarm_25050_labels = make_prediction(scores_25050, threshold, thresholding)

## Test Data(seen)
test_y_a = np.argmax(test_y_enc, axis=1)
test_seen_macro_f1 = f1_score(test_y_a, test_pred_simple, average='macro')
test_seen_acc = accuracy_score(test_y_a, test_pred_simple)
accuracy_score(test_y_a, test_pred_simple)
# print('Confusion Matrix(seen)')
# print(confusion_matrix(test_y_a, test_pred_simple))

## Total (Test & Unseen)
test_unseen_labels = np.concatenate([alarm_20052_labels,alarm_25050_labels])
test_pred = np.concatenate([test_pred_labels, test_unseen_labels])
test_true = np.concatenate([test_y_a.flatten(), np.ones_like(test_unseen_labels) * m])
test_macro_f1 = f1_score(test_true, test_pred, average='macro')
test_acc = accuracy_score(test_true, test_pred)
print('Confusion Matrix(overall)')
print(confusion_matrix(test_true, test_pred))
test_pred1 = np.concatenate([test_pred_labels, alarm_20052_labels])
test_true1 = np.concatenate([test_y_a.flatten(), np.ones_like(alarm_20052_labels) * m])
print('Confusion Matrix(overall)')
print(confusion_matrix(test_true1, test_pred1))

test_pred2 = np.concatenate([test_pred_labels, alarm_25050_labels])
test_true2 = np.concatenate([test_y_a.flatten(), np.ones_like(alarm_25050_labels) * m])
print('Confusion Matrix(overall)')
print(confusion_matrix(test_true2, test_pred2))

test_unseen_f1 = np.array([f1_score(np.ones_like(alarm_20052_labels), alarm_20052_labels == m),
                           f1_score(np.ones_like(alarm_25050_labels), alarm_25050_labels == m)])

test_unseen_accuracy = np.array([accuracy_score(np.ones_like(alarm_20052_labels), alarm_20052_labels == m),
                           accuracy_score(np.ones_like(alarm_25050_labels), alarm_25050_labels == m)])

print('overall f1: {:.4f}'.format(test_macro_f1))
print('overall acc: {:.4f}'.format(test_acc))
print('seen f1: {:.4f}'.format(test_seen_macro_f1))
print('seen acc: {:.4f}'.format(test_seen_acc))
print('unseen f1: {:.4f} / {:.4f}'.format(*test_unseen_f1))
print('unseen accuracy: {:.4f} / {:.4f}'.format(*test_unseen_accuracy))

