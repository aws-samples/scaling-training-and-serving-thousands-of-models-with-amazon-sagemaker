import os
import argparse
import pandas as pd

# Set up argument parser to parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-file", default="/opt/ml/processing/input/energy_consumption.csv", help="input CSV file")
parser.add_argument("--output-dir", default="/opt/ml/processing/output/customer_data", help="output directory")
args = parser.parse_args()

# Read input CSV file into Pandas DataFrame
df = pd.read_csv(args.input_file)

# Split data into one DataFrame per customer
customer_groups = df.groupby("customer_id")

# Iterate over customer groups
for customer_id, customer_df in customer_groups:
    
    # Drop customer_id column
    customer_df = customer_df.drop("customer_id", axis=1)
    
    # Save customer data to CSV file
    customer_output_path = os.path.join(args.output_dir, f"{str(customer_id)}.csv")
    customer_df.to_csv(customer_output_path, index=False, header=False)
