# Databricks notebook source
# MAGIC %run ./data/dataset

# COMMAND ----------

# MAGIC %run ./data/twitter

# COMMAND ----------

import tensorflow as tf

import datetime
import simplejson as json
import numpy as np

_DROPOUT_RATE = 0.95

dataset_name="imdb"
emb_dim=300
voc_size=380000
hid_dim=300
sen_len=300
batch_size=1000
epochs=50
twitter_train_file = "/dbfs/FileStore/tables/data/twitter_train.csv"
twitter_test_file = "/dbfs/FileStore/tables/data/twitter_test.csv"
word_dict_file = '/dbfs/FileStore/tables/data/imdb_word_index.json'
glove_file = '/dbfs/FileStore/tables/data/glove.6B.300d.txt'

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

input_layer = tf.keras.layers.Input(shape=(sen_len,), dtype=tf.int32)

layer_embed = tf.keras.layers.Embedding(voc_size, weights=[embedding_matrix], trainable=False, output_dim=emb_dim)(input_layer)
layer_lstm = tf.keras.layers.LSTM(1, dropout=0.95, return_sequences=True)(layer_embed)

layer_conv0 = tf.keras.layers.Conv1D(hid_dim, 6, activation="relu")(layer_embed)
layer_conv0 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv0)

layer_conv1 = tf.keras.layers.Conv1D(hid_dim, 5, activation="relu")(layer_embed)
layer_conv1 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv1)

layer_conv2 = tf.keras.layers.Conv1D(hid_dim, 4, activation="relu")(layer_embed)
layer_conv2 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv2)

layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer_embed)
layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)

layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 2, activation="relu")(layer_embed)
layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)

flatten = tf.keras.layers.Flatten()(layer_lstm)
layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3, layer_conv2, layer_conv1, layer_conv0, flatten], axis=1)
layer = tf.keras.layers.BatchNormalization()(layer)
layer = tf.keras.layers.Dropout(_DROPOUT_RATE)(layer)

output = tf.keras.layers.Dense(get_num_class(dataset_name), activation="softmax")(layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=output)
#load model weights if training stopped or crashed before
#model.load_weights(model_checkpoints)
model.summary()

# COMMAND ----------

tf.logging.info("Loading the data")
x_train_i, y_train_i, x_test_i, y_test_i = load(dataset_name, voc_size, sen_len)
x_train_t, y_train_t, x_test_t, y_test_t = load_twitter_data(twitter_train_file, twitter_test_file, voc_size, sen_len)

x_train = np.concatenate((x_train_i, np.array(x_train_t)))
y_train = np.concatenate((y_train_i, np.array(y_train_t)))
x_test = np.concatenate((x_test_i, np.array(x_test_t)))
y_test = np.concatenate((y_test_i, np.array(y_test_t)))

# COMMAND ----------

p = np.random.RandomState(seed=233).permutation(len(x_train))

# COMMAND ----------

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

now = datetime.datetime.now()
model_save_path = "sentiment_model_" + now.strftime("%Y%m%d%H%M") + ".h5"
model_checkpoints = model_save_path + "_weights.hdf5"
logdir="/tmp/ml/logs/" + model_save_path
checkpoints_path = "/tmp/ml/checkpoints/" + model_checkpoints

tf_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
tf_checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoints, monitor="val_acc", mode='max')

# COMMAND ----------

model.fit(x_train, y_train, batch_size=batch_size,
            validation_split=0.2, epochs=epochs, callbacks=[tf_callback, tf_checkpoints])
tf.keras.models.save_model(model, model_save_path, overwrite=True, include_optimizer=True)
score = model.evaluate(x_test, y_test, batch_size=batch_size)
tf.logging.info("Score: {}".format(score))

# COMMAND ----------

