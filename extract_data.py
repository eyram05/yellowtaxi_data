#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, row_number, monotonically_increasing_id, col
from pyspark.sql.window import Window
import psycopg2
from sqlalchemy import create_engine


# Set up a Spark session

spark = SparkSession.builder \
    .appName("Yellow Taxi Trip Data Transformation") \
    .config("spark.jars", "postgresql-42.7.3.jar") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()



file_path = "yellow_tripdata_2009-01DATASET (1).parquet"
df = spark.read.option("mode", "DROPMALFORMED").parquet(file_path)

df_subset = df.limit(10000)

# Find the missing values in the dataset

null_counts = [(column, df_subset.where(df_subset[column].isNull()).count()) for column in df_subset.columns]

columns_to_drops = [column for column, count in null_counts if count > 0.1 * df_subset.count()]


# Drop columns with more than 10% missing values
df_subset = df_subset.drop(*columns_to_drops)

null_counts = [(column, df_subset.where(df_subset[column].isNull()).count()) for column in df_subset.columns]

null_counts


# Data transformation 
df_subset = df_subset.filter(
    (df_subset["Passenger_Count"] > 0.0) &
    (df_subset["Trip_Distance"] > 0.0) &
    (df_subset["Fare_Amt"] > 0.0) &
    (df_subset["Total_Amt"] > 0.0) & 
    (df_subset["Tip_Amt"] >= 0.0) &
    (df_subset["Tolls_Amt"] >= 0.0) &
    (df_subset["surcharge"] >= 0.0)
)


# Convert the columns to the correct data types
columns_to_cast = {
    "Trip_Pickup_DateTime": "timestamp",
    "Trip_Dropoff_DateTime": "timestamp",
    "Passenger_Count": "integer",
}


for col_name, col_type in columns_to_cast.items():
    df_subset = df_subset.withColumn(col_name, df_subset[col_name].cast(col_type))



df_subset = df_subset.withColumn("Payment_Type", 
                                when(col("Payment_Type") == "CASH", "Cash")
                                .when(col("Payment_Type") == "CREDIT", "Credit")
                                .otherwise(col("Payment_Type")))



#NORMALIZATION#

vendors = df_subset.select("vendor_name").distinct() \
                    .withColumn("vendor_id", monotonically_increasing_id()) \
                    .select("vendor_id", "vendor_name")


locations = df_subset.select("Start_Lon", "Start_Lat", "End_Lon", "End_Lat") \
                    .withColumn("location_id", monotonically_increasing_id()) \
                    .select("location_id", "Start_Lon", "Start_Lat", "End_Lon", "End_Lat")


payments = df_subset.select("Trip_Pickup_DateTime", "Payment_Type", "Fare_Amt", "surcharge", "Tip_Amt", "Tolls_Amt", "Total_Amt" ) \
                    .withColumn("payment_id", monotonically_increasing_id()) \
                    .select("payment_id", "Payment_Type", "Fare_Amt", "surcharge", "Tip_Amt", "Tolls_Amt", "Total_Amt")


trips = df_subset.select("Trip_Pickup_DateTime", "Trip_Dropoff_DateTime", "Trip_Distance", "Passenger_Count",
                         "vendor_name", "Payment_Type", "Start_Lon") \
                         .withColumn("trip_id", monotonically_increasing_id()) \
                         .join(vendors, "vendor_name", "right") \
                         .select("trip_id", "vendor_id", "Trip_Pickup_DateTime", "Trip_Dropoff_DateTime", 
                                 "Trip_Distance", "Passenger_Count", "Payment_Type", "Start_Lon")


#the window function is to et a unique id for our trips table
windowSpec = Window.orderBy("trip_id")
trips = trips.withColumn("row_num", row_number().over(windowSpec))

trips_payments = trips.join(payments, trips.row_num == payments.payment_id, "inner") 

trips_payments_locations = trips_payments.join(locations, trips_payments.row_num == locations.location_id, "inner") 

trips_payments_locations.show()


trips_finals = trips_payments_locations.drop("row_num", "vendor_name", "Start_Lon", 
                                             "End_Lon", "Start_Lat", "End_Lat", "Payment_Type",
                                             "Fare_Amt", "surcharge", "Tip_Amt", "Tolls_Amt", "Total_Amt")


trips_finals.show()

#we have all the tables we need
#we need to save the data into a database

conn = psycopg2.connect(
    host="localhost",
    database="taxi_data",
    user="postgres",
    password="eyramSQL"
)

cur = conn.cursor()

cur.execute(
    """
    CREATE TABLE vendors (
        vendor_id INT PRIMARY KEY,
        vendor_name VARCHAR(50)
    );

    CREATE TABLE locations (
        location_id INT PRIMARY KEY,
        start_lon FLOAT,
        start_lat FLOAT,
        end_lon FLOAT,
        end_lat FLOAT
    );

    CREATE TABLE payments (
        payment_id INT PRIMARY KEY,
        payment_type VARCHAR(50),
        fare_amt FLOAT,
        surcharge FLOAT,
        tip_amt FLOAT,
        tolls_amt FLOAT,
        total_amt FLOAT
    );

     CREATE TABLE trips (
        trip_id INT PRIMARY KEY,
        trip_pickup_datetime TIMESTAMP,
        trip_dropoff_datetime TIMESTAMP,
        passenger_count INT,
        trip_distance FLOAT,
        vendor_id INT,
        location_id INT,
        payment_id INT,

        FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id),
        FOREIGN KEY (location_id) REFERENCES locations(location_id),
        FOREIGN KEY (payment_id) REFERENCES payments(payment_id)
    )
"""
)

conn.commit()

# Save the data into the database

dataframes_to_save = [
    (vendors, "vendors"),
    (locations, "locations"),
    (payments, "payments"),
    (trips_finals, "trips")
    ]

vendors_df = vendors.toPandas()
locations_df =locations.toPandas()
payments_df = payments.toPandas()
trips_df = trips_finals.toPandas()


from sqlalchemy import create_engine
connection_string = f'postgresql://postgres:eyramSQL@localhost/taxi_data'
engine = create_engine(connection_string)


trips_df.to_sql('trips', engine, if_exists='replace', index=False)
vendors_df.to_sql('vendors', engine, if_exists='replace', index=False)
payments_df.to_sql('payments', engine, if_exists='replace', index=False)
locations_df.to_sql('locations', engine, if_exists='replace', index=False)

conn.close()
print("Data saved Succesfully")
spark.close()
