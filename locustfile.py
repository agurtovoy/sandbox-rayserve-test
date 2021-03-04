import random
import os
import time
from locust import HttpUser, task, between

def log(str): print(str)

class ModelUser(HttpUser):
    wait_time = between(0.1, 1.5)

    @task()
    def call_model_endpoint(self):
        input = self._random_input()
        log(f"-> calling endpoint with {input}...")
        start_time = time.time()
        with open(os.path.join(self.input_dir, input), 'rb') as image:
            r = self.client.put("/detectron2/v1", data=image, headers={'Content-type': 'image/jpeg'})
            total_time = time.time() - start_time
            log(f"<- {input} response {r.status_code}: {r.headers['content-type'] if r.status_code == 200 else r.text} ({total_time:.2f} sec)")

    def on_start(self):
        random.seed()
        self.input_dir = os.path.join(os.path.dirname(__file__), "test/data/detectron2")
        self.inputs = os.listdir(self.input_dir)

    def _random_input(self):
        return random.choice(self.inputs)
