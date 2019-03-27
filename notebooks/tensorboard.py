# Databricks notebook source
dbutils.tensorboard.start("/dbfs/FileStore/ml/logs/")

# COMMAND ----------

dbutils.tensorboard.stop()

# COMMAND ----------

# MAGIC %sh
# MAGIC ls  /dbfs/FileStore/ml/logs/sentiment_model_201903220036.h5hvd_rank_1

# COMMAND ----------

