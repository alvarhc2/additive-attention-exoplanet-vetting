import time
import random
import pickle
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dropout

#@title get_train_val_test_tess
def get_train_val_test_tess():
    with open('tess_data/examples_train.pkl', "rb") as pkl_file:
        examples_train = pickle.load(pkl_file)
    with open('tess_data/examples_val.pkl', "rb") as pkl_file:
        examples_val = pickle.load(pkl_file)
    with open('tess_data/examples_test.pkl', "rb") as pkl_file:
        examples_test = pickle.load(pkl_file)
    trainX = [np.array([ex['time_series_features']['global_view'] for ex in examples_train]),
        np.array([ex['time_series_features']['local_view'] for ex in examples_train]),
        np.array([ex['time_series_features']['secondary_view'] for ex in examples_train]),
        np.array([ex['aux_features']['depth_change'] for ex in examples_train])
        ]
    trainy = np.array([ex['labels'] for ex in examples_train])

    valX = [np.array([ex['time_series_features']['global_view'] for ex in examples_val]),
        np.array([ex['time_series_features']['local_view'] for ex in examples_val]),
        np.array([ex['time_series_features']['secondary_view'] for ex in examples_val]),
        np.array([ex['aux_features']['depth_change'] for ex in examples_val])
        ]
    valy = np.array([ex['labels'] for ex in examples_val])

    testX = [np.array([ex['time_series_features']['global_view'] for ex in examples_test]),
        np.array([ex['time_series_features']['local_view'] for ex in examples_test]),
        np.array([ex['time_series_features']['secondary_view'] for ex in examples_test]),
        np.array([ex['aux_features']['depth_change'] for ex in examples_test])
        ]
    testy = np.array([ex['labels'] for ex in examples_test])
    return trainX, trainy, valX, valy, testX, testy

#@title divide_train_val_test_kepler
def divide_train_val_test_kepler(global_dataset, local_dataset, label_dataset):
    len_dataset = len(global_dataset)
    idcs = list(range(len_dataset))
    random.shuffle(idcs)
    print('SHUFFLED INDICES:')
    print(idcs)
    num_train = int(0.8 * len_dataset)
    num_val = int(0.1 * len_dataset)
    idcs_train = idcs[:num_train]
    idcs_val = idcs[num_train : num_train + num_val]
    idcs_test = idcs[num_train + num_val :]
    print(
        "Number training samples:",
        num_train,
        "Number validation/test samples:",
        num_val,
    )

    trainX_glob = [global_dataset[idx] for idx in idcs_train]
    trainX_loc = [local_dataset[idx] for idx in idcs_train]
    trainX = [np.array(trainX_glob), np.array(trainX_loc)]
    trainy = np.array([label_dataset[idx] for idx in idcs_train])

    valX_glob = [global_dataset[idx] for idx in idcs_val]
    valX_loc = [local_dataset[idx] for idx in idcs_val]
    valX = [np.array(valX_glob), np.array(valX_loc)]
    valy = np.array([label_dataset[idx] for idx in idcs_val])

    testX_glob = [global_dataset[idx] for idx in idcs_test]
    testX_loc = [local_dataset[idx] for idx in idcs_test]
    testX = [np.array(testX_glob), np.array(testX_loc)]
    testy = np.array([label_dataset[idx] for idx in idcs_test])

    return trainX, trainy, valX, valy, testX, testy

#@title load_dataset_kepler
def load_dataset_kepler():
    with open("kepler_data/"
    + "binned_light_curves.pkl", "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output[0], output[1], output[2]

#@title train_model_tess
def train_model_tess(model, trainX, trainy, valX, valy, learning_rate, epochs, batch_size, verbose, callbacks=None, class_weight=None):
    model.optimizer.lr = learning_rate
    start = time.time()
    # fit network
    history = model.fit(
        x=[trainX[0], trainX[1], trainX[2], trainX[3]],
        y=trainy,
        validation_data=([valX[0], valX[1], valX[2], valX[3]], valy),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        class_weight=class_weight
    )
    end = time.time()
    print("Training time:", end - start)
    return model, history

#@title train_model_kepler
def train_model_kepler(model, trainX, trainy, valX, valy, learning_rate, epochs, batch_size, verbose, callbacks):
    model.optimizer.lr = learning_rate
    start = time.time()
    # fit network
    model.fit(
        x=[trainX[0], trainX[1]],
        y=trainy,
        validation_data=([valX[0], valX[1]], valy),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )
    end = time.time()
    print("Training time:", end - start)
    return model

#@title save_history
def save_history(history):
    with open('saved_history.txt', 'a') as fsh:
        for key in history.history.keys():
            fsh.write(key + '\n')
            fsh.write(str(history.history[key]) + '\n')

#@title print_results
def print_results(testy, testy_pred):
  m = tf.keras.metrics.TruePositives()
  m.update_state(testy, testy_pred)
  print('tp:', m.result().numpy())
  m = tf.keras.metrics.FalsePositives()
  m.update_state(testy, testy_pred)
  print('fp:', m.result().numpy())
  m = tf.keras.metrics.TrueNegatives()
  m.update_state(testy, testy_pred)
  print('tn:', m.result().numpy())
  m = tf.keras.metrics.FalseNegatives()
  m.update_state(testy, testy_pred)
  print('fn:', m.result().numpy())
  m = tf.keras.metrics.BinaryAccuracy()
  m.update_state(testy, testy_pred)
  print('accuracy:', m.result().numpy())
  m = tf.keras.metrics.Precision()
  m.update_state(testy, testy_pred)
  print('precision:', m.result().numpy())
  m = tf.keras.metrics.Recall()
  m.update_state(testy, testy_pred)
  print('recall:', m.result().numpy())
  m = tf.keras.metrics.AUC()
  m.update_state(testy, testy_pred)
  print('auc:', m.result().numpy())

#@title attention
from keras.layers import Layer
from tensorflow.keras import backend as K
class attention(Layer):
    def __init__(self, verbose=0):
        super(attention,self).__init__()#**kwargs)
        self.verbose = verbose

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self,x):
        if self.verbose:
            print('\n******* Attention STARTS******')
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', x.shape)
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        if self.verbose:
            print('W : (, ) ', self.W.shape)
            print('b : (, ) ', self.b.shape)
            print('dot_product(values, W) : (, ) ', K.dot(x,self.W).shape)
            print('dot_product(values, W) + b : (, ) ', (K.dot(x,self.W)+self.b).shape)
            print('tanh(dot_product(values, W) + b) : (, ) ', K.tanh(K.dot(x,self.W)+self.b).shape)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        if self.verbose:
            print('squeeze : (, ) ', e.shape)
        # Compute the weights
        alpha = K.softmax(e)
        if self.verbose:
            print('attion_weights : (, ) ', alpha.shape)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        if self.verbose:
            print('attion_weights reshaped : (, ) ', alpha.shape)
        # Compute the context vector
        context = x * alpha
        if self.verbose:
            print('context_vector before reduce_sum: : (, ) ', context.shape)
        context = K.sum(context, axis=1)
        if self.verbose:
            print('context_vector with reduce_sum: : (, ) ', context.shape)
            print('\n******* Attention ENDS******')
        return context

#@title BahdanauAttention
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units, verbose=0):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.verbose = verbose

  def call(self, query, values):
    if self.verbose:
      print('\n******* Bahdanau Attention STARTS******')
      print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
      print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    if self.verbose:
      print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    if self.verbose:
      print('W1 x query_with_time_axis: (time axis, latent_space_dimension/units in W1) ', self.W1(query_with_time_axis).shape)
      print('W2 x values: (sequence length, latent_space_dimension/units in W2) ', self.W2(values).shape)
      print('sum: (sequence length, latent_space_dimension)', (self.W1(query_with_time_axis) + self.W2(values)).shape)
      print('tanh (sequence length, latent_space_dimension): ', tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)).shape)
      print('score: (batch_size, max_length, 1) ',score.shape)
    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)
    if self.verbose:
      print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    if self.verbose:
      print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    if self.verbose:
      print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)
      print('\n******* Bahdanau Attention ENDS******')
    return context_vector, attention_weights

#@title BahdanauAttentionMod
class BahdanauAttentionMod(tf.keras.layers.Layer):
  def __init__(self, units, verbose=0, **kwargs):
    super(BahdanauAttentionMod, self).__init__(**kwargs)
    self.units = units
    self.V = tf.keras.layers.Dense(1)
    # self.W2 = tf.keras.layers.Dense(self.units)
    self.verbose = verbose

  def build(self,input_shape):
    self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],self.units),
                               initializer='random_normal', trainable=True)
    self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],self.units),
                               initializer='zeros', trainable=True)
    super(BahdanauAttentionMod, self).build(input_shape)

  def call(self, values):
    if self.verbose:
      print('\n******* Bahdanau Attention STARTS******')
      print('values (encoder all hidden state): (time_steps, hidden_LSTM_size) ', values.shape)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    # score = self.V(tf.nn.tanh(self.b + self.W2(values)))
    # score = tf.nn.tanh(self.b + self.W2(values))
    score = self.V(tf.nn.tanh(self.b + K.dot(values,self.W)))
    if self.verbose:
      print('b: (time_steps, hidden_att_size)', self.b.shape)
      """
      print('values x W2: (time_steps, hidden_att_size)', self.W2(values).shape)
      print('b + (values x W2): (time_steps, hidden_att_size)', (self.b + self.W2(values)).shape)
      print('tanh(b + (values x W2)): (time_steps, hidden_att_size)', tf.nn.tanh(self.b + self.W2(values)).shape)
      """
      print('(tanh(b + (values x W2))) x V: (time_steps, hidden_V_size)', score.shape)
    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)
    if self.verbose:
      print('attention_weights: (time_steps, hidden_V_size)', attention_weights.shape)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    if self.verbose:
      print('context_vector before reduce_sum: (time_steps, hidden_LSTM_size) ',context_vector.shape)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    if self.verbose:
      print('context_vector after reduce_sum: (hidden_LSTM_size) ',context_vector.shape)
      print('\n******* Bahdanau Attention ENDS******')
    return context_vector, attention_weights

#@title get_model_cnn_kepler
def get_model_cnn_kepler(trainX, trainy, valX, valy, testX, testy, lr):
    verbose, epochs, batch_size = 1, 50, 64
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]

    IG = Input(shape=(n_timesteps_glob, n_features_glob), name="IG")
    CG1 = Conv1D(filters=16, kernel_size=5, activation="relu", name="CG1")(IG)
    CG2 = Conv1D(filters=16, kernel_size=5, activation="relu", name="CG2")(CG1)
    MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG2)
    CG3 = Conv1D(filters=32, kernel_size=5, activation="relu", name="CG3")(MG1)
    CG4 = Conv1D(filters=32, kernel_size=5, activation="relu", name="CG4")(CG3)
    MG2 = MaxPooling1D(pool_size=5, strides=2, name="MG2")(CG4)
    CG5 = Conv1D(filters=64, kernel_size=5, activation="relu", name="CG5")(MG2)
    CG6 = Conv1D(filters=64, kernel_size=5, activation="relu", name="CG6")(CG5)
    MG3 = MaxPooling1D(pool_size=5, strides=2, name="MG3")(CG6)
    CG7 = Conv1D(filters=128, kernel_size=5, activation="relu", name="CG7")(MG3)
    CG8 = Conv1D(filters=128, kernel_size=5, activation="relu", name="CG8")(CG7)
    MG4 = MaxPooling1D(pool_size=5, strides=2, name="MG4")(CG8)
    CG9 = Conv1D(filters=256, kernel_size=5, activation="relu", name="CG9")(MG4)
    CG10 = Conv1D(filters=256, kernel_size=5, activation="relu", name="CG10")(CG9)
    MG5 = MaxPooling1D(pool_size=5, strides=2, name="MG5")(CG10)
    MG5_flatten = Flatten()(MG5)

    IL = Input(shape=(n_timesteps_loc, n_features_loc), name="IL")
    CL1 = Conv1D(filters=16, kernel_size=5, activation="relu", name="CL1")(IL)
    CL2 = Conv1D(filters=16, kernel_size=5, activation="relu", name="CL2")(CL1)
    ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL2)
    CL3 = Conv1D(filters=32, kernel_size=5, activation="relu", name="CL3")(ML1)
    CL4 = Conv1D(filters=32, kernel_size=5, activation="relu", name="CL4")(CL3)
    ML2 = MaxPooling1D(pool_size=7, strides=2, name="ML2")(CL4)
    ML2_flatten = Flatten()(ML2)

    merged = Concatenate()([MG5_flatten, ML2_flatten])

    FC1 = Dense(512, activation="relu", name="FC1")(merged)
    FC2 = Dense(512, activation="relu", name="FC2")(FC1)
    FC3 = Dense(512, activation="relu", name="FC3")(FC2)
    FC4 = Dense(512, activation="relu", name="FC4")(FC3)
    OUT = Dense(1, activation="sigmoid", name="OUT")(FC4)

    model = Model(inputs=[IG, IL], outputs=[OUT])
    # plot_model(model, to_file='demo.png', show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title get_model_cnn_tess
def get_model_cnn_tess(trainX, lr):
    verbose, epochs = 1, 50
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]
    n_timesteps_sec, n_features_sec = trainX[2].shape[1], trainX[2].shape[2]
    n_features_dep = trainX[3].shape[1]

    IG = Input(shape=(n_timesteps_glob, n_features_glob), name="IG")
    CG1 = Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="CG1")(IG)
    CG2 = Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="CG2")(CG1)
    MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG2)
    CG3 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="CG3")(MG1)
    CG4 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="CG4")(CG3)
    MG2 = MaxPooling1D(pool_size=5, strides=2, name="MG2")(CG4)
    CG5 = Conv1D(filters=64, kernel_size=5, padding="same", activation="relu", name="CG5")(MG2)
    CG6 = Conv1D(filters=64, kernel_size=5, padding="same", activation="relu", name="CG6")(CG5)
    MG3 = MaxPooling1D(pool_size=5, strides=2, name="MG3")(CG6)
    CG7 = Conv1D(filters=128, kernel_size=5, padding="same", activation="relu", name="CG7")(MG3)
    CG8 = Conv1D(filters=128, kernel_size=5, padding="same", activation="relu", name="CG8")(CG7)
    MG4 = MaxPooling1D(pool_size=5, strides=2, name="MG4")(CG8)
    CG9 = Conv1D(filters=256, kernel_size=5, padding="same", activation="relu", name="CG9")(MG4)
    CG10 = Conv1D(filters=256, kernel_size=5, padding="same", activation="relu", name="CG10")(CG9)
    MG5 = MaxPooling1D(pool_size=5, strides=2, name="MG5")(CG10)
    MG5_flatten = Flatten()(MG5)

    IL = Input(shape=(n_timesteps_loc, n_features_loc), name="IL")
    CL1 = Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="CL1")(IL)
    CL2 = Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="CL2")(CL1)
    ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL2)
    CL3 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="CL3")(ML1)
    CL4 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="CL4")(CL3)
    ML2 = MaxPooling1D(pool_size=7, strides=2, name="ML2")(CL4)
    ML2_flatten = Flatten()(ML2)

    IS = Input(shape=(n_timesteps_sec, n_features_sec), name="IS")
    CS1 = Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="CS1")(IS)
    CS2 = Conv1D(filters=16, kernel_size=5, padding="same", activation="relu", name="CS2")(CS1)
    MS1 = MaxPooling1D(pool_size=7, strides=2, name="MS1")(CS2)
    CS3 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="CS3")(MS1)
    CS4 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="CS4")(CS3)
    MS2 = MaxPooling1D(pool_size=7, strides=2, name="MS2")(CS4)
    MS2_flatten = Flatten()(MS2)

    ID = Input(shape=(n_features_dep,), name="ID")
    # ID_flatten = Flatten()(ID)

    merged = Concatenate()([MG5_flatten, ML2_flatten, MS2_flatten, ID])

    FC1 = Dense(512, activation="relu", name="FC1")(merged)
    FC2 = Dense(512, activation="relu", name="FC2")(FC1)
    FC3 = Dense(512, activation="relu", name="FC3")(FC2)
    FC4 = Dense(512, activation="relu", name="FC4")(FC3)
    OUT = Dense(1, activation="sigmoid", name="OUT")(FC4)

    # sigmoid_cross_entropy_with_logits / sparse_softmax_cross_entropy_with_logits / compute_weighted_loss?

    model = Model(inputs=[IG, IL, IS, ID], outputs=[OUT])
    print(model.summary())
    # plot_model(model, to_file='demo.png', show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title get_model_cnn_lstm_kepler
def get_model_cnn_lstm_kepler(trainX, trainy, valX, valy, testX, testy, lr):
    verbose, epochs, batch_size = 1, 50, 64
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]

    ing = Input(shape=(n_timesteps_glob, n_features_glob), name="ing")
    cg1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg1")(ing)
    cg2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg2")(cg1)
    # MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG1)
    lg1 = LSTM(500, activation="tanh", return_sequences=True, name="lg1")(cg2)
    lg2 = LSTM(200, activation="tanh", return_sequences=True, name="lg2")(lg1)
    lg3 = LSTM(100, activation="tanh", return_sequences=True, name="lg3")(lg2)
    atg = attention()(lg3)

    inl = Input(shape=(n_timesteps_loc, n_features_loc), name="inl")
    cl1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl1")(inl)
    cl2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl2")(cl1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ll1 = LSTM(200, activation="tanh", return_sequences=True, name="ll1")(cl2)
    ll2 = LSTM(100, activation="tanh", return_sequences=True, name="ll2")(ll1)
    ll3 = LSTM(50, activation="tanh", return_sequences=True, name="ll3")(ll2)
    atl = attention()(ll3)

    merged = Concatenate()([atg, atl])

    fc1 = Dense(150, activation="relu", name="fc1")(merged)
    fc2 = Dense(150, activation="relu", name="fc2")(fc1)
    fc3 = Dense(150, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="out")(fc3)

    model = Model(inputs=[ing, inl], outputs=[out])
    # plot_model(model, to_file="cnn_lstm_2.png", show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title get_model_cnn_lstm_kepler_bahd_mod
def get_model_cnn_lstm_kepler_bahd_mod(trainX, trainy, valX, valy, testX, testy, lr):
    verbose, epochs, batch_size = 1, 50, 64
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]

    ing = Input(shape=(n_timesteps_glob, n_features_glob), name="ing")
    cg1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg1")(ing)
    cg2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg2")(cg1)
    # MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG1)
    lg1 = LSTM(500, activation="tanh", return_sequences=True, name="lg1")(cg2)
    lg2 = LSTM(200, activation="tanh", return_sequences=True, name="lg2")(lg1)
    lg3 = LSTM(100, activation="tanh", return_sequences=True, name="lg3")(lg2)
    atg_context_vector, atg_attention_weights = BahdanauAttentionMod(1, verbose=0)(lg3)

    inl = Input(shape=(n_timesteps_loc, n_features_loc), name="inl")
    cl1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl1")(inl)
    cl2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl2")(cl1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ll1 = LSTM(200, activation="tanh", return_sequences=True, name="ll1")(cl2)
    ll2 = LSTM(100, activation="tanh", return_sequences=True, name="ll2")(ll1)
    ll3 = LSTM(50, activation="tanh", return_sequences=True, name="ll3")(ll2)
    atl_context_vector, atl_attention_weights = BahdanauAttentionMod(1, verbose=0)(ll3)

    merged = Concatenate()([atg_context_vector, atl_context_vector])

    fc1 = Dense(150, activation="relu", name="fc1")(merged)
    fc2 = Dense(150, activation="relu", name="fc2")(fc1)
    fc3 = Dense(150, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="out")(fc3)

    model = Model(inputs=[ing, inl], outputs=[out])
    print(model.summary())
    # plot_model(model, to_file="cnn_lstm_2.png", show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title get_model_cnn_lstm_tess_att
def get_model_cnn_lstm_tess_att(trainX, lr):
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]
    n_timesteps_sec, n_features_sec = trainX[2].shape[1], trainX[2].shape[2]
    n_features_dep = trainX[3].shape[1]

    ing = Input(shape=(n_timesteps_glob, n_features_glob), name="ing")
    cg1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg1")(ing)
    cg2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg2")(cg1)
    # MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG1)
    lg1 = LSTM(500, activation="tanh", return_sequences=True, name="lg1")(cg2)
    lg2 = LSTM(200, activation="tanh", return_sequences=True, name="lg2")(lg1)
    lg3 = LSTM(100, activation="tanh", return_sequences=True, name="lg3")(lg2)
    atg = attention(verbose=0)(lg3)

    inl = Input(shape=(n_timesteps_loc, n_features_loc), name="inl")
    cl1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl1")(inl)
    cl2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl2")(cl1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ll1 = LSTM(200, activation="tanh", return_sequences=True, name="ll1")(cl2)
    ll2 = LSTM(100, activation="tanh", return_sequences=True, name="ll2")(ll1)
    ll3 = LSTM(50, activation="tanh", return_sequences=True, name="ll3")(ll2)
    atl = attention(verbose=0)(ll3)

    ins = Input(shape=(n_timesteps_sec, n_features_sec), name="ins")
    cs1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs1")(ins)
    cs2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs2")(cs1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ls1 = LSTM(200, activation="tanh", return_sequences=True, name="ls1")(cs2)
    ls2 = LSTM(100, activation="tanh", return_sequences=True, name="ls2")(ls1)
    ls3 = LSTM(50, activation="tanh", return_sequences=True, name="ls3")(ls2)
    ats = attention(verbose=0)(ls3)

    ind = Input(shape=(n_features_dep), name="ind")

    merged = Concatenate()([atg, atl, ats, ind])

    fc1 = Dense(150, activation="relu", name="fc1")(merged)
    fc2 = Dense(150, activation="relu", name="fc2")(fc1)
    fc3 = Dense(150, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="out")(fc3)

    model = Model(inputs=[ing, inl, ins, ind], outputs=[out])
    print(model.summary())
    # plot_model(model, to_file="cnn_lstm.png", show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title get_model_cnn_lstm_tess_bahd
def get_model_cnn_lstm_tess_bahd(trainX, lr):
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]
    n_timesteps_sec, n_features_sec = trainX[2].shape[1], trainX[2].shape[2]
    n_features_dep = trainX[3].shape[1]

    ing = Input(shape=(n_timesteps_glob, n_features_glob), name="ing")
    cg1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg1")(ing)
    cg2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg2")(cg1)
    # MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG1)
    lg1 = LSTM(500, activation="tanh", return_sequences=True, name="lg1")(cg2)
    lg2 = LSTM(200, activation="tanh", return_sequences=True, name="lg2")(lg1)
    lg3_outputs, lg3_state_h, lg3_state_c = LSTM(100, activation="tanh", return_sequences=True, return_state=True, name="lg3")(lg2)
    atg_context_vector, atg_attention_weights = BahdanauAttention(1, verbose=0)(lg3_state_h, lg3_outputs)

    inl = Input(shape=(n_timesteps_loc, n_features_loc), name="inl")
    cl1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl1")(inl)
    cl2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl2")(cl1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ll1 = LSTM(200, activation="tanh", return_sequences=True, name="ll1")(cl2)
    ll2 = LSTM(100, activation="tanh", return_sequences=True, name="ll2")(ll1)
    ll3_outputs, ll3_state_h, ll3_state_c = LSTM(50, activation="tanh", return_sequences=True, return_state=True, name="ll3")(ll2)
    atl_context_vector, atl_attention_weights = BahdanauAttention(1, verbose=0)(ll3_state_h, ll3_outputs)

    ins = Input(shape=(n_timesteps_sec, n_features_sec), name="ins")
    cs1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs1")(ins)
    cs2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs2")(cs1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ls1 = LSTM(200, activation="tanh", return_sequences=True, name="ls1")(cs2)
    ls2 = LSTM(100, activation="tanh", return_sequences=True, name="ls2")(ls1)
    ls3_outputs, ls3_state_h, ls3_state_c = LSTM(50, activation="tanh", return_sequences=True, return_state=True, name="ls3")(ls2)
    ats_context_vector, ats_attention_weights = BahdanauAttention(1, verbose=0)(ls3_state_h, ls3_outputs)

    ind = Input(shape=(n_features_dep), name="ind")

    #print(atg_context_vector.shape)
    #print(atl_context_vector.shape)
    #print(ats_context_vector.shape)
    #print(ind.shape)
    merged = Concatenate()([atg_context_vector, atl_context_vector, ats_context_vector, ind])

    fc1 = Dense(150, activation="relu", name="fc1")(merged)
    fc2 = Dense(150, activation="relu", name="fc2")(fc1)
    fc3 = Dense(150, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="out")(fc3)

    model = Model(inputs=[ing, inl, ins, ind], outputs=[out])
    print(model.summary())
    # plot_model(model, to_file="cnn_lstm.png", show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title get_model_cnn_lstm_tess_bahd_mod
def get_model_cnn_lstm_tess_bahd_mod(trainX, lr):
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]
    n_timesteps_sec, n_features_sec = trainX[2].shape[1], trainX[2].shape[2]
    n_features_dep = trainX[3].shape[1]

    ing = Input(shape=(n_timesteps_glob, n_features_glob), name="ing")
    cg1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg1")(ing)
    cg2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg2")(cg1)
    # MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG1)
    lg1 = LSTM(500, activation="tanh", return_sequences=True, name="lg1")(cg2)
    lg2 = LSTM(200, activation="tanh", return_sequences=True, name="lg2")(lg1)
    lg3 = LSTM(100, activation="tanh", return_sequences=True, name="lg3")(lg2)
    atg_context_vector, atg_attention_weights = BahdanauAttentionMod(1, verbose=0)(lg3)

    inl = Input(shape=(n_timesteps_loc, n_features_loc), name="inl")
    cl1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl1")(inl)
    cl2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl2")(cl1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ll1 = LSTM(200, activation="tanh", return_sequences=True, name="ll1")(cl2)
    ll2 = LSTM(100, activation="tanh", return_sequences=True, name="ll2")(ll1)
    ll3 = LSTM(50, activation="tanh", return_sequences=True, name="ll3")(ll2)
    atl_context_vector, atl_attention_weights = BahdanauAttentionMod(1, verbose=0)(ll3)

    ins = Input(shape=(n_timesteps_sec, n_features_sec), name="ins")
    cs1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs1")(ins)
    cs2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs2")(cs1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ls1 = LSTM(200, activation="tanh", return_sequences=True, name="ls1")(cs2)
    ls2 = LSTM(100, activation="tanh", return_sequences=True, name="ls2")(ls1)
    ls3 = LSTM(50, activation="tanh", return_sequences=True, name="ls3")(ls2)
    ats_context_vector, ats_attention_weights = BahdanauAttentionMod(1, verbose=0)(ls3)

    ind = Input(shape=(n_features_dep,), name="ind")

    #print(atg_context_vector.shape)
    #print(atl_context_vector.shape)
    #print(ats_context_vector.shape)
    #print(ind.shape)
    merged = Concatenate()([atg_context_vector, atl_context_vector, ats_context_vector, ind])

    fc1 = Dense(150, activation="relu", name="fc1")(merged)
    fc2 = Dense(150, activation="relu", name="fc2")(fc1)
    fc3 = Dense(150, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="out")(fc3)

    model = Model(inputs=[ing, inl, ins, ind], outputs=[out])
    print(model.summary())
    # plot_model(model, to_file="cnn_lstm.png", show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

def get_model_cnn_lstm_tess(trainX, lr):
    n_timesteps_glob, n_features_glob = trainX[0].shape[1], trainX[0].shape[2]
    n_timesteps_loc, n_features_loc = trainX[1].shape[1], trainX[1].shape[2]
    n_timesteps_sec, n_features_sec = trainX[2].shape[1], trainX[2].shape[2]
    n_features_dep = trainX[3].shape[1]

    ing = Input(shape=(n_timesteps_glob, n_features_glob), name="ing")
    cg1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg1")(ing)
    cg2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cg2")(cg1)
    # MG1 = MaxPooling1D(pool_size=5, strides=2, name="MG1")(CG1)
    lg1 = LSTM(500, activation="tanh", return_sequences=True, name="lg1")(cg2)
    lg2 = LSTM(200, activation="tanh", return_sequences=True, name="lg2")(lg1)
    lg3 = LSTM(100, activation="tanh", return_sequences=False, name="lg3")(lg2)
    # atg_context_vector, atg_attention_weights = BahdanauAttentionMod(1, verbose=0)(lg3)

    inl = Input(shape=(n_timesteps_loc, n_features_loc), name="inl")
    cl1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl1")(inl)
    cl2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cl2")(cl1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ll1 = LSTM(200, activation="tanh", return_sequences=True, name="ll1")(cl2)
    ll2 = LSTM(100, activation="tanh", return_sequences=True, name="ll2")(ll1)
    ll3 = LSTM(50, activation="tanh", return_sequences=False, name="ll3")(ll2)
    # atl_context_vector, atl_attention_weights = BahdanauAttentionMod(1, verbose=0)(ll3)

    ins = Input(shape=(n_timesteps_sec, n_features_sec), name="ins")
    cs1 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs1")(ins)
    cs2 = Conv1D(filters=128, kernel_size=10, activation="relu", name="cs2")(cs1)
    # ML1 = MaxPooling1D(pool_size=7, strides=2, name="ML1")(CL1)
    ls1 = LSTM(200, activation="tanh", return_sequences=True, name="ls1")(cs2)
    ls2 = LSTM(100, activation="tanh", return_sequences=True, name="ls2")(ls1)
    ls3 = LSTM(50, activation="tanh", return_sequences=False, name="ls3")(ls2)
    # ats_context_vector, ats_attention_weights = BahdanauAttentionMod(1, verbose=0)(ls3)

    ind = Input(shape=(n_features_dep,), name="ind")

    #print(atg_context_vector.shape)
    #print(atl_context_vector.shape)
    #print(ats_context_vector.shape)
    #print(ind.shape)
    merged = Concatenate()([lg3, ll3, ls3, ind])

    fc1 = Dense(150, activation="relu", name="fc1")(merged)
    fc2 = Dense(150, activation="relu", name="fc2")(fc1)
    fc3 = Dense(150, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="out")(fc3)

    model = Model(inputs=[ing, inl, ins, ind], outputs=[out])
    print(model.summary())
    # plot_model(model, to_file="cnn_lstm.png", show_shapes=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

#@title
def divide_train_val_test_concat_kepler(global_dataset, local_dataset, label_dataset):
    len_dataset = len(global_dataset)
    idcs = list(range(len_dataset))
    random.shuffle(idcs)
    num_train = int(0.8 * len_dataset)
    num_val = int(0.1 * len_dataset)
    idcs_train = idcs[:num_train]
    idcs_val = idcs[num_train : num_train + num_val]
    idcs_test = idcs[num_train + num_val :]
    print(
        "Number training samples:",
        num_train,
        "Number validation/test samples:",
        num_val,
    )

    trainX = [global_dataset[idx] + local_dataset[idx] for idx in idcs_train]
    trainX = np.array(trainX)
    trainy = np.array([label_dataset[idx] for idx in idcs_train])

    valX = [global_dataset[idx] + local_dataset[idx] for idx in idcs_val]
    valX = np.array(valX)
    valy = np.array([label_dataset[idx] for idx in idcs_val])

    testX = [global_dataset[idx] + local_dataset[idx] for idx in idcs_test]
    testX = np.array(testX)
    testy = np.array([label_dataset[idx] for idx in idcs_test])

    return trainX, trainy, valX, valy, testX, testy

def concat_tess(trainX, trainy, valX, valy, testX, testy):
    new_depth_change = np.expand_dims(trainX[3], axis=-1)
    new_trainX = np.concatenate((trainX[0], trainX[1], trainX[2], new_depth_change), axis=1)

    new_depth_change = np.expand_dims(valX[3], axis=-1)
    new_valX = np.concatenate((valX[0], valX[1], valX[2], new_depth_change), axis=1)

    new_depth_change = np.expand_dims(testX[3], axis=-1)
    new_testX = np.concatenate((testX[0], testX[1], testX[2], new_depth_change), axis=1)

    return new_trainX, trainy, new_valX, valy, new_testX, testy

def train_model_concat_kepler(model, trainX, trainy, valX, valy, learning_rate, epochs, batch_size, verbose, callbacks):
    model.optimizer.lr = learning_rate
    start = time.time()
    # fit network
    history = model.fit(
        x=trainX,
        y=trainy,
        validation_data=(valX, valy),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )
    end = time.time()
    print("Training time:", end - start)
    return model, history

def train_model_concat_tess(model, trainX, trainy, valX, valy, learning_rate, epochs, batch_size, verbose, callbacks):
    model.optimizer.lr = learning_rate
    start = time.time()
    # fit network
    history = model.fit(
        x=trainX,
        y=trainy,
        validation_data=(valX, valy),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )
    end = time.time()
    print("Training time:", end - start)
    return model, history

def get_model_lstm_dense_attention_kepler(trainX, trainy, valX, valy, testX, testy, lr):
    verbose, epochs, batch_size = 1, 50, 64
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]

    inp = Input(shape=(n_timesteps, n_features), name="inp")
    lstm = LSTM(1, activation="tanh", return_sequences=True)(inp)
    at_context_vector, at_attention_weights = BahdanauAttentionMod(1, verbose=0)(lstm)
    dens = Dense(1, activation="sigmoid")(at_context_vector)
    model = Model(inputs=[inp], outputs=[dens])
    print(model.summary())

    """
    model = Sequential()
    model.add(LSTM(1, activation="tanh", return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(attention())
    model.add(Dense(1, activation="sigmoid"))
    """

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

def get_model_lstm_dense_attention_tess(trainX, lr):
    verbose, epochs, batch_size = 1, 50, 64
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]

    inp = Input(shape=(n_timesteps, n_features), name="inp")
    lstm = LSTM(1, activation="tanh", return_sequences=False)(inp)#return_sequences=True)(inp)
    # at_context_vector, at_attention_weights = BahdanauAttentionMod(1, verbose=0)(lstm)
    dens = Dense(1, activation="sigmoid")(lstm)#(at_context_vector)
    model = Model(inputs=[inp], outputs=[dens])
    print(model.summary())

    """
    model = Sequential()
    model.add(LSTM(1, activation="tanh", return_sequences=True, input_shape=(n_timesteps, n_features)))
    # model.add(BahdanauAttentionMod(1, verbose=0))
    model.add(Dense(1, activation="sigmoid"))
    print(model.summary())
    """

    opt = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model
