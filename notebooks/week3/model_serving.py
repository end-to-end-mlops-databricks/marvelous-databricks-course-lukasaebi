# Databricks notebook source
# %pip install -q ../../reservations-0.0.1-py3-none-any.whl

# COMMAND ----------
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, Route, ServedEntityInput, TrafficConfig
from pyspark.sql import SparkSession

from reservations.config import DataConfig
from reservations.data import DataLoader, DataPreprocessor

# COMMAND ----------
workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
config = DataConfig.from_yaml(config_path="../../data/config.yaml")
catalog_name = config.catalog_name
schema_name = config.schema_name
volume_name = config.volume_name
table_name = "hotel_reservations"
endpoint_name = "hotel-reservations-model-serving"

# COMMAND ----------
workspace.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel_reservations_model",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=3,
            )
        ],
        # Optional if only 1 entity is served
        traffic_config=TrafficConfig(
            routes=[Route(served_model_name="hotel_reservations_model-3", traffic_percentage=100)]
        ),
    ),
)

# COMMAND ----------
endpoint = workspace.serving_endpoints.get(name=endpoint_name)
print(endpoint.state)

# COMMAND ----------
# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
token = "some-token"
host = spark.conf.get("spark.databricks.workspaceUrl")
model_serving_endpoint = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"

# COMMAND ----------
data = spark.table(f"{catalog_name}.{schema_name}.{table_name}").toPandas()

data_preprocessor = DataPreprocessor(config)
X_encoded, y_encoded = data_preprocessor.preprocess_data(X=data, target=config.target)

dataloader = DataLoader(config)
X_train, X_test, y_train, y_test = dataloader.split_data(X=X_encoded, y=y_encoded)

# COMMAND ----------
X_test_sample = X_test.sample(n=500, replace=True).to_dict(orient="records")
inference_data = [[record] for record in X_test_sample]

# COMMAND ----------
response = requests.post(
    model_serving_endpoint,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": inference_data[0]},
)
