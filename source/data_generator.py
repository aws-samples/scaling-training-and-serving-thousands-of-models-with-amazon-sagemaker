import random
import numpy as np
import pandas as pd
import os


def generate_data(customer_id, start_date, end_date):
    # Generate random baseline consumption value
    baseline = random.uniform(500, 1000)
    # Generate random amplitude for sinusoidal seasonality
    amplitude = random.uniform(100, 200)
    # Set period of sinusoidal seasonality to 1 year (365 days)
    period = 365
    # Generate random phase shift for sinusoidal seasonality
    phase_shift = random.uniform(0, 2*np.pi)

    def seasonality(t):
        # Calculate sinusoidal seasonality value based on input timestamp
        return amplitude * np.sin(2*np.pi*t/period + phase_shift)

    # Calculate consumption value as sum of baseline and sinusoidal seasonality
    consumption = baseline + \
        seasonality(pd.date_range(start_date, end_date).astype(int))
    # Generate timestamps for each day in the specified date range
    timestamp = pd.date_range(start_date, end_date)
    # Return data as Pandas DataFrame
    return pd.DataFrame({"customer_id": customer_id, "timestamp": timestamp, "consumption": consumption})


def main(begin_timestamp, end_timestamp, num_customers, output_file, output_dir, seed=42):
    # Set random seed
    random.seed(seed)
    # Generate data for each customer
    data = [generate_data(customer, begin_timestamp, end_timestamp)
            for customer in range(num_customers)]
    # Concatenate data into single DataFrame
    df = pd.concat(data)
    # Save data to CSV file
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    # return output path
    print(f'Data created successfully at {output_path}.')
    return output_path


if __name__ == "__main__":
    import argparse

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--begin_timestamp', type=str,
                        default="2022-01-01 00:00:00")
    parser.add_argument('--end_timestamp', type=str,
                        default="2022-12-31 23:59:59")
    parser.add_argument('--num_customers', type=int, default=1000)
    parser.add_argument('--output_file', type=str,
                        default="energy_consumption.csv")
    parser.add_argument('--output_dir', type=str, default="../data")
    args = parser.parse_args()

    # Execution
    output_path = main(args.seed, args.begin_timestamp, args.end_timestamp, args.num_customers,
         args.output_file, args.output_dir)
   
