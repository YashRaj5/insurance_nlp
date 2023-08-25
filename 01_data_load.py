# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="800px"/>

# COMMAND ----------

# MAGIC
# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC While companies across industries have accelerated digital adoption after the COVID-19 pandemic, are insurers meeting the ever-changing demands from customers?
# MAGIC  
# MAGIC As an insurer, are you spending most of your time and resources into **creating business value**?
# MAGIC
# MAGIC **Customer service** is a vital part of the insurance business. This is true for multiple business cases: from **marketing**, to **customer retention**, and **claims**. 
# MAGIC
# MAGIC At the same time, turnover in customer service teams is significantly higher than others, while training such teams takes time and effort. What's more, the fact that insurance companies frequently outsource customer service to different players also represents a challenge in terms of **service quality** and **consistency**.
# MAGIC
# MAGIC By **digitalizing** these processes, insurers can seamlessly:
# MAGIC  
# MAGIC * **Increase customer satisfaction** by **reducing waiting times**
# MAGIC * Provide a better, **interactive experience** by reducing amount of phone calls
# MAGIC * Reduce their phone bill costs
# MAGIC * **Scale** their operations by being able to do more with less
# MAGIC * Shift **money** and **human resources** from **operational** processes to actual **product** and **value creation**.
# MAGIC
# MAGIC This solutions accelerator is a head start on developing and deploying a **machine learning solution** to detect customer intents based on pieces of unstructured text from an **Interactive Voice Response (IVR)** stream, or from a **virtual agent** - which could be integrated with a mobile app, SMS, Whatsapp and communication channels.
# MAGIC  
# MAGIC ## Target Solution
# MAGIC
# MAGIC <img src="https://github.com/rafaelvp-db/dbx-insurance-qa-hugging-face/blob/master/img/Insurance(1).png?raw=true">

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

# MAGIC %run "./00_config"

# COMMAND ----------

# DBTITLE 1,Downloading the insurance QA Dataset
from datasets import load_dataset

dataset = load_dataset("j0selit0/insurance-qa-en")

# COMMAND ----------

display(dataset['train'].to_pandas().loc[:10, ["question_en", "topic_en"]])

# COMMAND ----------

# DBTITLE 1,Basic cleaning
# let's conver everything to lower case and remove extra spaces
import re
from datasets import ClassLabel

# COMMAND ----------

def clean(example: str) -> str:
 
  output = []
  for question in example["question_en"]:
    question_clean = question.lower()
    question_clean = re.sub(' {2,}', ' ', question_clean)
    output.append(question_clean)
  
  example["question_en"] = output
  return example

# COMMAND ----------

clean_dataset = dataset.map(lambda example: clean(example), batched=True)

# COMMAND ----------

# Renaming our column and converting labels to ClassLabel
clean_dataset = clean_dataset.remove_columns(["index"])
clean_dataset = clean_dataset.rename_columns({"question_en": "text", "topic_en": "label"})
names = list(set(clean_dataset["train"]["label"]))
clean_dataset = clean_dataset.cast_column("label", ClassLabel(names = names))

# COMMAND ----------

# Save to cleaned dataset for further training
## We first save the clean dataset to a local path in the driver, then copy it to DBFS because pyarrow is unable to write directly to DBFS
local_path = "/tmp/insuranceqa"
dbutils.fs.rm(config["main_path"], True)
clean_dataset.save_to_disk(local_path)
dbutils.fs.cp(f"file:///{local_path}", config["main_path"], recurse = True)

# COMMAND ----------

# DBTITLE 1,Generating a data profile
# By using the display function, we can easily generate a data profile for our dataset
 
display(dataset["train"].to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC * Looking at the profile above, we can notice that questions related to life insurance are quite frequent. As an insurance company, we might be interested into taking actions to leverage this aspect within our audience - for instance, marketing, sales or educational campaigns. On the other hand, it could also be that our customer service team needs to be enabled or scaled. We will look at the wider distribution of topics/intents in order to look for more insights.

# COMMAND ----------

display(dataset["train"].to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC * Looking at the distribution of topics/intents across our training set, we can see that apart from life insurance, auto insurance and medicare are quite popular themes.
# MAGIC * When we look at the opposite direction - less popular intents, we can highlight critical illness insurance, long term care insurance and other insurance. Here, we might also be interested in understanding more about these specific liness of businesses, and even compare profit margins across them. The fact that there are few questions around these topics could also mean that we are doing a better job at enabling customers to solve their problems or answer their questions through digital channels, without having to talk to a human agent.

# COMMAND ----------

# MAGIC %md
# MAGIC Here we create the table by converting a HuggingFace dataset to Pandas, then to write as a Delta table. This may have scalability contraints as the dataset cannot be larger than your driver memory.
# MAGIC
# MAGIC In more realistic scenarios, your data may already come in formats that Spark can process in parallelized ways.

# COMMAND ----------

# DBTITLE 1,Saving the test set into Delta for inference
test_df = spark.createDataFrame(dataset["test"].to_pandas())
test_df.write.saveAsTable("questions", mode="overwrite")
