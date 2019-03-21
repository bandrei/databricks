# Databricks notebook source
# MAGIC %run ./data/dataset

# COMMAND ----------

# MAGIC %run ./data/twitter

# COMMAND ----------

import tensorflow as tf

import datetime
import simplejson as json
import numpy as np
import horovod.tensorflow.keras as hvd
import math

_DROPOUT_RATE = 0.95

dataset_name="imdb"
emb_dim=300
voc_size=380000
hid_dim=300
sen_len=300
batch_size=1000
epochs=80
twitter_train_file = "/dbfs/mnt/data/twitter_train.csv"
twitter_test_file = "/dbfs/mnt/data/twitter_test.csv"
word_dict_file = '/dbfs/mnt/data/imdb_word_index.json'
glove_file = '/dbfs/mnt/data/glove.6B.300d.txt'

# COMMAND ----------

with open(word_dict_file) as f:
    word_index = json.load(f)

embeddings_index = dict()
with open(glove_file) as gl:
    for line in gl:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# COMMAND ----------

embedding_matrix = np.zeros((voc_size, emb_dim))
for word, i in word_index.items():
    if i < voc_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#apple_index = word_index['apple']
#print(embeddings_index.get('apple'))

# COMMAND ----------

tf.logging.info("Loading the data")
x_train_i, y_train_i, x_test_i, y_test_i = load(dataset_name, voc_size, sen_len)
x_train_t, y_train_t, x_test_t, y_test_t = load_twitter_data(twitter_train_file, twitter_test_file, voc_size, sen_len)

x_train = np.concatenate((x_train_i, np.array(x_train_t)))
y_train = np.concatenate((y_train_i, np.array(y_train_t)))
x_test = np.concatenate((x_test_i, np.array(x_test_t)))
y_test = np.concatenate((y_test_i, np.array(y_test_t)))

p = np.random.RandomState(seed=233).permutation(len(x_train))

# COMMAND ----------

def train(epochs, logdir, checkpoints_path, model_path):
  hvd.init()
  # Horovod: adjust number of epochs based on number of GPUs.
  epochs = int(math.ceil(epochs / hvd.size()))

  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(hvd.local_rank())
  
  input_layer = tf.keras.layers.Input(shape=(sen_len,), dtype=tf.int32)

  layer_embed = tf.keras.layers.Embedding(voc_size, weights=[embedding_matrix], trainable=True, output_dim=emb_dim)(input_layer)
  layer_lstm = tf.keras.layers.LSTM(1, dropout=0.95, return_sequences=True)(layer_embed)

  layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer_embed)
  layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)

  layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 2, activation="relu")(layer_embed)
  layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)

  flatten = tf.keras.layers.Flatten()(layer_lstm)
  layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3, flatten], axis=1)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(_DROPOUT_RATE)(layer)

  output = tf.keras.layers.Dense(get_num_class(dataset_name), activation="softmax")(layer)

  model = tf.keras.models.Model(inputs=[input_layer], outputs=output)
  #load model weights if training stopped or crashed before
  #model.load_weights(model_checkpoints)
  model.summary()
  
  opt = tf.keras.optimizers.RMSprop()

  opt = hvd.DistributedOptimizer(opt)

  model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

  callbacks = []
  callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
  print("HVD rank {}".format(hvd.rank()))
  if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, monitor="val_acc", mode='max'))
    
  model.fit(x_train[:70000,:], y_train[:70000,:], batch_size=batch_size, validation_data=(x_train[70000:90000,:],y_train[70000:90000,:]), epochs=epochs, callbacks=callbacks)
  tf.keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)

# COMMAND ----------

def score():
  score = model.evaluate(x_test, y_test, batch_size=batch_size)
  tf.logging.info("Score: {}".format(score))

# COMMAND ----------

from sparkdl import HorovodRunner
hr = HorovodRunner(np=2)
now = datetime.datetime.now()
model_name = "sentiment_model_" + now.strftime("%Y%m%d%H%M") + ".h5"
model_checkpoints = model_name + "_weights.hdf5"
logdir="/dbfs/FileStore/ml/logs" + model_name
checkpoints_path = "/dbfs/mnt/models/" + model_checkpoints
model_save_path = "/dbfs/mnt/models/" + model_name
hr.run(train, epochs=80, logdir=logdir, model_path=model_save_path, checkpoints_path=checkpoints_path)
score()

# COMMAND ----------

