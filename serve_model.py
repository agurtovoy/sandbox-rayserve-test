import argparse
import ray
from ray import serve
from model import build_model_class

from collections import namedtuple
Endpoint = namedtuple('Endpoint', ['model_name', 'model_version', 'options', 'methods'])

def model_id(endpoint):
    return f"{endpoint.model_name}_{endpoint.model_version}"

def backend_id(model_id): return f"model.{model_id}"
def endpoint_id(model_id): return f"endpoint.{model_id}"

def delete_endpoint(client, endpoint):
    id = model_id(endpoint)
    client.delete_endpoint(endpoint_id(id))
    client.delete_backend(backend_id(id))

def create_endpoint(client, endpoint):
    id = model_id(endpoint)
    client.create_backend(backend_id(id), build_model_class(endpoint.model_name, endpoint.model_version), ray_actor_options=endpoint.options)
    client.create_endpoint(endpoint_id(id), backend=backend_id(id), route=f"/{endpoint.model_name}/{endpoint.model_version}", methods=endpoint.methods)


def client_init(endpoints, update):
    if update:
        client = serve.connect()
        for endpoint in endpoints:
            delete_endpoint(client, endpoint)
        return client

    # `detached=True` starts a long-running Ray Serve instance service, see also `ray_init()`
    return serve.start(http_host="0.0.0.0", http_port=8000, detached=True)

def ray_init():
    # `address="auto"`` means we're connecting to a long-lived Ray cluster, assumes we started one prior to running
    # this script; see https://docs.ray.io/en/master/serve/deployment.html#lifetime-of-a-ray-serve-instance
    ray.init(address="auto") 

def main(**args):
    ray_init()
    endpoints = [
        Endpoint("detectron2", "v1", { "num_gpus": 1 }, ["PUT"]),
        Endpoint("echo", "v1", None, ["PUT"]),
    ]

    client = client_init(endpoints, **args)
    for endpoint in endpoints:
        create_endpoint(client, endpoint)

def parse_args():
    parser = argparse.ArgumentParser(description="Serve pointrend model through Ray Serve.")
    parser.add_argument("-u", dest="update", action="store_true", help="Update the model's endpoint")
    return vars(parser.parse_args())

main(**parse_args())
