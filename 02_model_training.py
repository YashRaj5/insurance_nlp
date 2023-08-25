# Databricks notebook source
# MAGIC %pip install -q datasets

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from transformers import pipeline

pipe = pipeline("text-classification")
pipe(["This restaurant is awesome", "This restaurant is awful"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine Tuning our Model
# MAGIC * We need to create a dataset in a format which is acceptable by Hugging Face
# MAGIC * We need to define how our data will be encoded or tokenized
# MAGIC * Our model must have 12 different labels; we will leverage the `AutoModelForSequenceClassification` class from Hugging Face to customise that part

# COMMAND ----------

# DBTITLE 1,Reading the data
import datasets
 
local_path = "/tmp/insurance"
dbutils.fs.rm(f"file://{local_path}", recurse = True)
dbutils.fs.cp(config["main_path"], f"file://{local_path}", recurse = True)
dataset = datasets.load_from_disk(local_path)

# COMMAND ----------

# DBTITLE 1,Dataset tokenization
from transformers import AutoTokenizer
 
base_model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# COMMAND ----------

def tokenize(examples):
  return tokenizer(examples["text"], padding = True, truncation = True, return_tensors = "pt")

# COMMAND ----------

tokenized_dataset = dataset.map(tokenize, batched=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training our Model
# MAGIC Here we will:
# MAGIC
# MAGIC * Create a `Trainer` object - this is a helper class from Hugging Face which makes training easier
# MAGIC * Instantiate a `TrainingArguments` object
# MAGIC * Create an `EarlyStoppingCallback` - this will help us avoid our model overfit
# MAGIC * Train our model

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
import datasets

# COMMAND ----------

dataset = datasets.load_from_disk("/dbfs/tmp/insurance")
label2id = dataset["train"].features["label"]._str2int
id2label = dataset["train"].features["label"]._int2str

# COMMAND ----------

model = AutoModelForSequenceClassification.from_pretrained(
  base_model,
  num_labels = len(label2id),
  label2id = label2id,
  id2label = dict(enumerate(id2label))
)

# COMMAND ----------

model
