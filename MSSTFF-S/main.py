# _*_ coding: UTF-8 _*_
import sklearn
import tensorflow as tf

from scipy.stats import wasserstein_distance
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow_core.python.keras import Input

config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))
import torch
from sklearn.metrics import f1_score, accuracy_score,matthews_corrcoef
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from swa.tfkeras import SWA


def get_distance_measure(x, p=1):
    N, D = x.shape

    dist = tf.tile(x, [1, N])
    dist = tf.transpose(dist)

    dist = tf.pow(tf.abs(dist - tf.transpose(dist)), p)
    dist = tf.pow(dist, 1.0 / p)

    return dist


def paa(series: np.array, emb_size: int, scaler=None):


    series_len = len(series)
    if scaler:
        series = series / scaler

    if (series_len == emb_size):
        return np.copy(series)
    else:
        for i in range(0, emb_size * series_len):
            idx = i // series_len
            pos = i // emb_size
            np.add.at(res, idx, series[pos])
            # res[idx] = res[idx] + series[pos]
        return res / series_len


def multi_dimension_paa(series: np.array, emb_size: int):

    paa_out = np.zeros((series.shape[0], emb_size))

    for k in range(series.shape[0]):
        paa_out[k] = paa(series[k].flatten(), emb_size)
    return paa_out


def create_distance_similarity_matrix(series: np.array, emb_size: int, p: int):



    if series.shape[1] < series.shape[0]:
        series = series.T

    series = multi_dimension_paa(series, emb_size)
    series = series.T
    series = torch.tensor(series).float()
    dist = get_distance_measure(series, p)
    return dist.unsqueeze(0)



def create_N_distance_similarity_matrix(series, emb_size, p):
    d, T = series.shape
    dist = np.empty((d, emb_size, emb_size), dtype=np.float32)

    for k in tqdm(range(d), total=d):
        series_tf = tf.convert_to_tensor(series[k, :], dtype=tf.float32)
        dist[k] = create_distance_similarity_matrix_tf(series_tf, emb_size, p).numpy()

    return dist


from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Bidirectional, LSTM, Conv2D, Reshape, \
    MaxPooling2D, GRU,Conv1D,GlobalAvgPool2D,GlobalAvgPool1D
from keras.utils import plot_model
from tensorflow.keras.models import Model
from keras.utils import np_utils
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
#rom scipy.stats import wasserstein_distance
import convnext

def show_acc(history):
    plt.clf()
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)



class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.2):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.query_dense = Dense(self.d_model)
        self.key_dense = Dense(self.d_model)
        self.value_dense = Dense(self.d_model)

        self.combine_heads = Dense(self.d_model)

        self.dropout = Dropout(dropout_rate)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query = inputs
        key = inputs
        value = inputs
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.combine_heads(concat_attention)
        output = self.dropout(output)

        return output

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights





def _create_awrgcnnblstm(emb_size, labelnum):




    # inpt = Input(shape=(emb_size,emb_size, 1))
    # input = tf.reshape(inpt, [-1, emb_size, emb_size])
    # x = GRU(256, return_sequences=True)(input)
    # x1 = GRU(256, return_sequences=True)(x)
    # x = GRU(256)(x1)
    # x = Dense(256)(x)
    # x = MultiHeadSelfAttention(8,64)(x)
    # x = Dense(64)(x)
    # x = tf.concat([x1, x], axis=-1)
    # x2 = tf.keras.layers.LayerNormalization()(x)
    # x = Dense(64,activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    # x = tf.concat([x2, x], axis=-1)
    # x = tf.keras.layers.LayerNormalization()(x)
    # x = Dense(labelnum,activation="softmax")(x)
    # model = Model(inputs=inpt, outputs=x)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # return model




    inpt = Input(shape=(emb_size, emb_size, 1))


    x1 = convnext.Block(dim=32)(inpt)
    x2 = convnext.DownsampleLayer(64)(x1)
    x2 = convnext.Block(dim=64)(x2)
    x3 = convnext.DownsampleLayer(64)(x2)
    x3 = convnext.Block(dim=32)(x3)

    input = tf.reshape(inpt, [-1, emb_size, emb_size])
    y1 = Bidirectional(GRU(32, return_sequences=True))(input)
    y2 = Bidirectional(GRU(32, return_sequences=True))(y1)
    y3 = Bidirectional(GRU(32, return_sequences=True))(y2)

    X1 = GlobalAvgPool2D()(x1)
    X2 = GlobalAvgPool2D()(x2)
    X3 = GlobalAvgPool2D()(x3)

    Y1 = GlobalAvgPool1D()(y1)
    Y2 = GlobalAvgPool1D()(y2)
    Y3 = GlobalAvgPool1D()(y3)

    r1 = tf.concat([X1, Y1], axis=-1)
    r2 = tf.concat([X2, Y2], axis=-1)
    r3 = tf.concat([r1, r2, X3, Y3], axis=-1)
    x = Dense(labelnum, activation="softmax")(r3)
    model = Model(inputs=inpt, outputs=x)
    #opt = Nadam()
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.summary()
    return model



import sys


class Logger(object):
    def __init__(self, fileN="Default.log"):
            self.terminal = sys.stdout
            self.log = open(fileN, "a")

    def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
            pass


# from swa.tfkeras import SWA

# model.summary()
def AWRG_CNN_BLSTMdisaggregator(X_train, X_test, Y_train, train_factory, device, emb_size):
    X_train = np.reshape(X_train, (X_train.shape[0], emb_size, emb_size, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], emb_size, emb_size, 1))
    Y_train = np_utils.to_categorical(Y_train).astype(int)
    labelnum = Y_train.shape[1]
    model = _create_awrgcnnblstm(emb_size, labelnum)

    # 统计模型的参数量
    flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
    print("Total FLOPs: {:.2f} FLOPs".format(flops.total_float_ops ))

    early_stopping = EarlyStopping(monitor='accuracy', patience=10)

    epoch = 150
    start_epoch = 128

    # # define swa callback
    swa = SWA(start_epoch=start_epoch,
              lr_schedule='constant',
              swa_lr=0.001,
              swa_lr2=0.005,
              swa_freq=3,
              batch_size=32,  # needed when using batch norm
              verbose=1)


    start = time.time()
    print("========== TRAIN ============")
    history = model.fit(X_train, Y_train, epochs=epoch, verbose=1, batch_size=32, shuffle=False,
                        validation_split=0.0588)  # , callbacks=[early_stopping], validation_split=1/16checkpoint,#,class_weight=cw #,sample_weight=sample_weights #callbacks=[early_stopping]
    # print("CHECKPOINT {}".format(epochs))

    show_acc(history)
    # show_loss(history)

    end = time.time()
    print("Train =", end - start, "seconds.")

    print("========== DISAGGREGATE ============")
    # make predictions
    Y_pred = model.predict(X_test)
    # loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128) # 评估模型
    # Y_test,Y_pred onehot encoding
    # test=np.argmax(Y_test, axis=-1)
    pred = np.argmax(Y_pred, axis=1)
    # test,pred label encoding
    return pred


if __name__ == '__main__':
    emb_size = 50
    eps = 10
    delta = 10
    factory = {1: [0]}
    print("========== OPEN DATASETS ============")
    Result = []
    for train_factory in factory.keys():
        test_factory = train_factory
        for device in factory[train_factory]:

            # 读取数据
            # xpower = np.load('.npy')
            # ylabel = np.load('.npy')



            xpower = xpower.values[:, 1:]
            #xpower = np.reshape(xpower, (-1, 300))
            ylabel = ylabel.values[:, 1:]



            Y = ylabel[:, device]
            Y = np.array(Y, dtype=np.int64)

            X = xpower
            X = (X - X.min()) / (X.max() - X.min())
            dist = create_N_distance_similarity_matrix(X, emb_size, p=1)
            #         dist = torch.floor(dist*eps)
            #         dist[dist>delta]=delta
            distarray = dist.numpy()
            X = distarray
            # splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1, shuffle=False)  # ,stratify=Y



            Y_pred = AWRG_CNN_BLSTMdisaggregator(X_train, X_test, Y_train, train_factory, device, emb_size)



            F1 = f1_score(Y_test, Y_pred, average='micro')
            mcc = matthews_corrcoef(Y_test, Y_pred)
            D1 = wasserstein_distance(Y_test, Y_pred)
            Accuracy = accuracy_score(Y_test, Y_pred)
            print([train_factory, device, Accuracy, F1, mcc, D1])
            print([train_factory, device, Accuracy, F1, mcc, D1])
            Result.append([train_factory, device, Accuracy, F1, mcc, D1])









