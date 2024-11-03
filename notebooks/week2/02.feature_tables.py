# Databricks notebook source

# COMMAND ----------
from datetime import datetime

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from reservations.config import DataConfig
from reservations.data import DataLoader, DataPreprocessor
from reservations.evaluate import Accuracy
from reservations.model import RandomForestModel

# Initialize the Databricks session and clients
workspace = WorkspaceClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------
config = DataConfig.from_yaml(config_path="../../data/config.yaml")

# Extract configuration details
num_features = config.numerical_variables
cat_features = config.categorical_variables
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name
volume_name = config.volume_name

# Define table names and function name
table_name = "hotel_reservations_features"
feature_table_name = f"{catalog_name}.{schema_name}.{table_name}"
function_name = f"{catalog_name}.{schema_name}.calculate_years_since_booking"


# COMMAND ----------
# Load data
spark = SparkSession.builder.getOrCreate()
df = spark.table(f"{catalog_name}.{schema_name}.hotel_reservations").toPandas()

# COMMAND ----------
# Create or replace the hotel_reservations_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.{table_name}
(Booking_ID STRING NOT NULL,
 no_of_children INT,
 required_car_parking_space INT,
 repeated_guest INT,
 no_of_special_requests INT)
""")

# Add primary key constraint
spark.sql(f"""
ALTER TABLE {catalog_name}.{schema_name}.{table_name}
ADD CONSTRAINT reservation_pk PRIMARY KEY (Booking_ID)
""")

# Enable Change Data Feed
spark.sql(f"""
ALTER TABLE {catalog_name}.{schema_name}.{table_name}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Insert data into the feature table from both train and test sets
spark.sql(f"""
INSERT INTO {catalog_name}.{schema_name}.{table_name}
SELECT Booking_ID, no_of_children, required_car_parking_space, repeated_guest, no_of_special_requests
FROM {catalog_name}.{schema_name}.hotel_reservations
""")

# COMMAND ----------
# Define a function to calculate the years since booking
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(arrival_year BIGINT)
RETURNS INT
LANGUAGE PYTHON AS
$$
from datetime import datetime
return datetime.now().year - arrival_year
$$
""")
# COMMAND ----------
df = spark.table(f"{catalog_name}.{schema_name}.hotel_reservations").toPandas()
dataloader = DataLoader(config)
train, test = dataloader.split_data(df)
preprocessor = DataPreprocessor(config)

train_set = spark.createDataFrame(
    train.drop(columns=["no_of_children", "required_car_parking_space", "repeated_guest", "no_of_special_requests"])
)

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["no_of_children", "required_car_parking_space", "repeated_guest", "no_of_special_requests"],
            lookup_key="Booking_ID",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="years_since_booking",
            input_bindings={"arrival_year": "arrival_year"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate years_since_booking for test set
current_year = datetime.now().year
test["years_since_booking"] = current_year - test["arrival_year"]

# add years_since_booking to config
config.numerical_variables.append("years_since_booking")

# Preprocess data
preprocessor = DataPreprocessor(config)
X_train, y_train = preprocessor.preprocess_data(training_df, target)
X_test, y_test = preprocessor.preprocess_data(test, target)

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Users/lukas.aebi@axpo.com/week2_experiment")

with mlflow.start_run(tags={"branch": "week2", "git_sha": "test"}) as run:
    run_id = run.info.run_id
    model = RandomForestModel()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    acc = Accuracy.calculate(y_test, y_preds)
    mlflow.log_metric("accuracy", acc)

    signature = infer_signature(model_input=X_train, model_output=y_preds)

    # Log model with feature engineering
    fe.log_model(
        model=model,
        flavor=mlflow.pyfunc,
        artifact_path="random-forest-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f"runs:/{run_id}/random-forest-model-fe", name=f"{catalog_name}.{schema_name}.random_forest_model_fe"
)
