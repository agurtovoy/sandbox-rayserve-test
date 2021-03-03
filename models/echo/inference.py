import logging
import json
import time

logger = logging.getLogger("model")

def model_fn():
    pass

def input_fn(request_body, request_content_type):
    return json.loads(request_body)

def predict_fn(input, model):    
    logger.info(f"running predictions on input {type(input)}")
    try:
        sleep = input["sleep"]
        if sleep is not None:
            time.sleep(sleep)

        return { "input": input }
    except Exception as e:
        logger.error("prediction failed:", e)
        raise

def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
