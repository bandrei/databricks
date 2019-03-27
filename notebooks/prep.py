# Databricks notebook source
# MAGIC %sh 
# MAGIC sudo wget http://nlp.stanford.edu/data/glove.6B.zip -O /dbfs/tmp/glove.zip

# COMMAND ----------

# MAGIC %sh 
# MAGIC unzip /dbfs/tmp/glove.zip -d /dbfs/mnt/data/

# COMMAND ----------

# MAGIC %sh
# MAGIC cp /dbfs/FileStore/tables/data/twitter_train.csv /dbfs/mnt/data/
# MAGIC cp /dbfs/FileStore/tables/data/twitter_test.csv /dbfs/mnt/data/
# MAGIC cp /dbfs/FileStore/tables/data/imdb_word_index.json /dbfs/mnt/data/