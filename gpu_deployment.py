# File name: gpu_deployment.py
# This file deploys a translator application and a driver that calls it.
# The translator application uses a pre-trained model from the transformers library.
# It specifies that the deployment should use 0.2 GPU cores.

from starlette.requests import Request

import ray
from ray import serve
from ray.serve.deployment_graph import RayServeDAGHandle

from transformers import pipeline

import os

# Creates a Ray Serve deployment for a translator application.
# It specifies that the deployment should use 0.2 GPU cores.
# Refer to https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html# for more information.
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.2})
class Translator:
    def __init__(self):
        # Load model and set device_map to "auto" to automatically detect GPU availability
        self.model = pipeline("translation_en_to_fr", model="t5-small", device_map="auto")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    # Asynchronously calls the translate function.
    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        translation = self.translate(english_text)
        return translation

# Creates a Ray Serve deployment for a driver that calls the translator application.
@serve.deployment(ray_actor_options={"num_gpus": 0.2})
class BasicDriver:
    def __init__(self, dag: RayServeDAGHandle):
        self.dag = dag

    # Asynchronously calls the translator application.
    async def __call__(self, http_request: Request):
        object_ref = await self.dag.remote(http_request)
        result = await object_ref
        return result

# Binds the translator application and the driver to the same deployment.
translator_app = Translator.bind()
DagNode = BasicDriver.bind(translator_app)