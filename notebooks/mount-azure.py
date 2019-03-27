# Databricks notebook source
#mount models folder
dbutils.fs.mount(
  source = "wasbs://models@mlmessaging.blob.core.windows.net",
  mount_point = "/mnt/models/",
  extra_configs = {"fs.azure.account.key.mlmessaging.blob.core.windows.net":dbutils.secrets.get(scope = "mltraining", key = "mlmessaging-blob-key")})

# COMMAND ----------

#mount data folder
dbutils.fs.mount(
  source = "wasbs://data@mlmessaging.blob.core.windows.net",
  mount_point = "/mnt/data/",
  extra_configs = {"fs.azure.account.key.mlmessaging.blob.core.windows.net":dbutils.secrets.get(scope = "mltraining", key = "mlmessaging-blob-key")})

# COMMAND ----------

databricks secrets list-acls --scope mltraining

# COMMAND ----------

