import ast
import json
import os
from prophet.serialize import model_from_json

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
        data = int(ast.literal_eval(data)['period'])
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
