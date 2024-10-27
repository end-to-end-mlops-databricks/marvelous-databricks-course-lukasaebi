# Databricks notebook source

import logging
from datetime import datetime, timezone

import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
import pandas as pd
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from reservations.config import DataConfig
from reservations.data import DataLoader
from reservations.evaluate import Accuracy
from reservations.model import RandomForestModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------
catalog_name = "axe_lab_playground"
schema_name = "mlops_course"
volume_name = "additional_data"
config = DataConfig.from_yaml(config_path="../data/config.yaml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
df = spark.table(f"{catalog_name}.{schema_name}.hotel_reservations").toPandas()
dataloader = DataLoader(df, config)
X_train, X_test, y_train, y_test = dataloader.preprocess_data()

# COMMAND ----------
mlflow.set_experiment("/Users/lukas.aebi@axpo.com/week2_experiment")
with mlflow.start_run(
    run_name=now,
    tags={"branch": "week2"},
) as run:
    run_id = run.info.run_id

    model = RandomForestModel()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    acc = Accuracy.calculate(y_test, y_preds)
    mlflow.log_metric("accuracy", acc)
    train = pd.concat([X_train, y_train], axis=1)
    dataset = mlflow.data.pandas_dataset.from_pandas(
        df=train,
        source=f"{catalog_name}.{schema_name}.hotel_reservations",
        name="hotel_reservations",
        targets=config.target,
    )
    mlflow.log_input(dataset, context="training")

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.pyfunc.log_model(
        python_model=model,
        artifact_path="random-forest-model",
        code_paths=["./reservations-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/random-forest-model")
loaded_model.unwrap_python_model()

# COMMAND ----------
model_name = f"{catalog_name}.{schema_name}.hotel_reservations_model"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/random-forest-model", name=model_name, tags={"git_sha": "test"}
)

# COMMAND ----------
model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)
