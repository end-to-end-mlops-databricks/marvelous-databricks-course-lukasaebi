# Databricks notebook source
# MAGIC %pip install ../../housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------

import mlflow
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from reservations.config import DataConfig

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Load config
config = DataConfig.from_yaml(config_path="../../config.yml")

# Get feature columns details
num_features = config.numerical_variables
cat_features = config.categorical_variables
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.hotel_reservations_preds"
online_table_name = f"{catalog_name}.{schema_name}.hotel_reservations_preds_online"

# Load training and test sets from Catalog
df = spark.table(f"{catalog_name}.{schema_name}.hotel_reservations").toPandas()

# COMMAND ----------
model = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.hotel_reservations_model/3")

# COMMAND ----------
preds_df = df[
    ["Booking_ID", "no_of_children", "required_car_parking_space", "repeated_guest", "no_of_special_requests"]
]
preds_df["predicted_booking_status"] = model.predict(df[cat_features + num_features])

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------
# 1. Create a feature table
fe.create_table(
    name=feature_table_name,
    primary_keys=["Booking_ID"],
    df=preds_df,
    description="Hotel reservations predictions feature table.",
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------
# 2. Create the online table using the feature table

spec = OnlineTableSpec(
    primary_key_columns=["Booking_ID"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------
# 3. Create feature spec containing feature lookup

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="Booking_ID",
        feature_names=["no_of_children", "required_car_parking_space", "repeated_guest", "no_of_special_requests"],
    )
]

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------
# 4. Create endpoing using feature spec

serving_endpoint_name = "hotel-reservations-feature-serving"

# Create a feature serving endpoint
workspace.serving_endpoints.create(
    name=serving_endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,
                scale_to_zero_enabled=True,
                workload_size="Small",
            )
        ]
    ),
)

# COMMAND ----------
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # type: ignore  # noqa: F821
host = spark.conf.get("spark.databricks.workspaceUrl")

serving_endpoint_url = f"https://{host}/serving-endpoints/{serving_endpoint_name}/invocations"

response = requests.post(
    serving_endpoint_url,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Id": "1"}]},
)
