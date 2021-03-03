import logging

import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

logger = logging.getLogger("model")


def _load_d2_config():
    result = get_cfg()
    result.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    result.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    result.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return result

def model_fn():
    d2_config = _load_d2_config()
    result = DefaultPredictor(d2_config)
    return result


def input_fn(request_body, request_content_type):
    result = np.asarray(cv2.imdecode(np.frombuffer(request_body, np.uint8), cv2.IMREAD_COLOR))
    logger.info(f"converted input image to {type(result)} of shape {result.shape}")
    return result


def predict_fn(input, model):    
    result = model(input)["instances"].to("cpu")
    logger.info(f"prediction result: {type(result)}")
    return (result, input, model.cfg)


def _image_to_jpeg(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    result, encoded_image = cv2.imencode('.jpg', image, encode_param)
    if not result: raise Exception('Error converting raw image to JPEG')
    return encoded_image.tobytes()

def to_jpeg_output(result):
    logger.debug('converting output to JPEG')
    (model_output, input, cfg) = result
    v = Visualizer(input[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(model_output)
    return _image_to_jpeg(out.get_image()[:, :, ::-1])

_output_serializers = {
    "image/jpeg": to_jpeg_output
}

def output_fn(result, response_content_type):
    serializer = _output_serializers.get(response_content_type)
    if serializer is None: 
        raise Exception(f"Unsupported response content type {response_content_type}")

    return serializer(result)
