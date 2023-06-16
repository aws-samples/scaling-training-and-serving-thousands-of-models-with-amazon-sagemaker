import os
import argparse
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import tarfile
import glob
import json


def model_fn(model_dir):
    """Load the prophet model from the model_dir. This is the same model
    that is saved in the main if statement.
    """
    model_json = open(os.path.join(model_dir, 'model.json'), 'r')
    model = model_from_json(model_json.read())
    print("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    # Deserialize the input data
    if request_content_type == 'application/json':
        # Load the data as a JSON object
        data = json.loads(request_body)
        data = int(data['period'])
        return data
    else:
        raise ValueError(
            'Unsupported content type: {}'.format(request_content_type))
    


def predict_fn(input_data: int, model):
    """For the input data, run prediction using the loaded model."""
    # Put the data into a dataframe
    future = model.make_future_dataframe(periods=input_data)
    # Make predictions
    forecast = model.predict(future)
    return forecast[-input_data:]


def output_fn(prediction, accept):
    # Serialize the predictions
    if accept == 'application/json':
        # Serialize the predictions as a JSON object
        prediction = prediction.to_json()
        response_body = json.dumps(prediction)
    else:
        raise ValueError('Unsupported accept type: {}'.format(accept))

    # Return the prediction properly encoded
    return response_body



if __name__ == '__main__':

    # Set up argument parser to parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default=os.environ["SM_CHANNEL_TRAINING"], help="input directory")
    parser.add_argument(
        "--output-dir", default="/opt/ml/checkpoints", help="output directory")
    parser.add_argument("--code-dir", default="/opt/ml/code",
                        help="code directory")
    parser.add_argument("--prediction-length", type=int,
                        default=7, help="prediction length")
    parser.add_argument("--seasonality-mode",
                        default="multiplicative", help="seasonality mode")
    parser.add_argument("--yearly-seasonality",
                        action="store_true", help="enable yearly seasonality")
    parser.add_argument("--weekly-seasonality",
                        action="store_true", help="enable weekly seasonality")
    parser.add_argument("--daily-seasonality",
                        action="store_true", help="enable daily seasonality")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Print all the files
    customer_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    print(f"[{os.environ['SM_CURRENT_HOST']}] List of customer files: {customer_files}")

    # Iterate over customer files in input directory
    for input_file in customer_files:

        # Get the customer ID from the filename
        customer = input_file.split("/")[-1].split(".")[0]

        # Read the CSV file into a data frame
        df = pd.read_csv(input_file, names=["timestamp", "consumption"])

        # Convert timestamp column to DatetimeIndex
        df.index = pd.to_datetime(df.timestamp)

        # Drop timestamp column
        df = df.drop("timestamp", axis=1)

        # Split data into training and test sets
        train_df, test_df = df[:-
                               args.prediction_length], df[-args.prediction_length:]

        # Initialize Prophet model
        model = Prophet(seasonality_mode=args.seasonality_mode, yearly_seasonality=args.yearly_seasonality,
                        weekly_seasonality=args.weekly_seasonality, daily_seasonality=args.daily_seasonality)

        # Rename columns to fit Prophet's requirements
        train_df = train_df.reset_index().rename(
            columns={"timestamp": "ds", "consumption": "y"})

        # Fit model to training data
        model.fit(train_df)

        # Save model to output directory
        with open('/tmp/model.json', 'w') as fout:
            fout.write(model_to_json(model))

        # Create the model.tar.gz archive containing the model and the training script
        with tarfile.open(os.path.join(args.output_dir, f'{customer}.tar.gz'), "w:gz") as tar:
            tar.add('/tmp/model.json', "model.json")
            tar.add(os.path.join(args.code_dir, "training.py"),
                    "code/inference.py")
            tar.add(os.path.join(args.code_dir, "requirements.txt"),
                    "code/requirements.txt")
        print(f"[{os.environ['SM_CURRENT_HOST']}] Saved model to {os.path.join(args.output_dir, f'{customer}.tar.gz')}")
