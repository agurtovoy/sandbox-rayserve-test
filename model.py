import sys
import importlib
import time
import json

from starlette.responses import Response
from utils import configureLogger

logger = configureLogger("model")


def error_code(e):
    if isinstance(e, AssertionError): return 400
    if isinstance(e, ValueError): return 422
    return 500

def error_response(e):
    code = error_code(e)
    json_body = { "error": { "code": code, "message": str(e) }}
    return Response(json.dumps(json_body), status_code=code, media_type="application/json")

class Model:
    def __init__(self, name, version, response_content_type=None):
        self.name = name
        self.version = version
        self.response_content_type = response_content_type

        self.fns = importlib.import_module(f"models.{name}.inference")
        logger.debug(f"imported model module")

        self.model = self._load_model()
        logger.info("constructed the model")

    async def __call__(self, request):
        logger.info(f">> calling model {self.name}/{self.version}")
        start_time = time.time()
        content_type = request.headers["content-type"]
        try:
            return Response(
                self.process(await request.body(), content_type), 
                media_type=self._response_type(content_type)
                )
        except Exception as e:
            return error_response(e)
        finally:
            total_time = time.time() - start_time
            logger.info(f"<< model {self.name}/{self.version} took {total_time:.2f} sec to run")


    def process(self, data, content_type):
        model_input = self._preprocess(data, content_type)
        model_out = self._predict(model_input)
        return self._postprocess(model_out, content_type)

    def _load_model(self):
        logger.info(f"loading model")
        try:
            return self.fns.model_fn()
        except Exception as e:
            logger.error(f"failed to load the model: {e}")
            raise

    def _preprocess(self, data, content_type):
        logger.info(f"preprocessing input data of type {content_type}")
        try:
            return self.fns.input_fn(data, content_type)
        except Exception as e:
            logger.error("failed to preprocess input data:", e)
            raise

    def _predict(self, input):
        logger.info(f"running {type(self.model)} on input {type(input)}")
        try:
            return self.fns.predict_fn(input, self.model)
        except Exception as e:
            logger.error("prediction failed:", e)
            raise

    def _postprocess(self, output, content_type):
        response_content_type = self._response_type(content_type)
        logger.info(f"postprocessing + serializing model output of type {type(output)} to {response_content_type}")
        try:
            result = self.fns.output_fn(output, response_content_type)
            logger.debug(f"succesfully serialized model output to {type(result)}")
            return result
        except Exception as e:
            logger.error("postprocessing failed:", e)
            raise


    def _response_type(self, content_type):
        return self.response_content_type if self.response_content_type is not None else content_type


def build_model_class(name, version, response_content_type=None):
    def init_model(self):
        logger.debug(f"constructing model {name}, version {version}")
        super(type(self), self).__init__(name, version, response_content_type)

    return type(name, (Model,), { "__init__": init_model })


if __name__ == '__main__':
    def read_image(filename):
        with open(filename, "rb") as f:
            return f.read()

    def write_image(filename, output):
        with open(filename, "wb") as f:
            f.write(output)

    Model = build_model_class("detectron2", "v1")
    m = Model()
    output = m.process(read_image(sys.argv[1]), "image/jpeg")
    write_image(sys.argv[2], output)
