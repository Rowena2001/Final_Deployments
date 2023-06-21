# File name: helloworld.py
# This file deploys a simple "hello world" function and a driver that calls it.
# It follows the basic_dag.py example created by the ray-project team.
# Refer to https://github.com/ray-project/test_dag for original source code.

from ray import serve
from ray.serve.deployment_graph import RayServeDAGHandle

# Creates a Ray Serve deployment for a simple "hello world" function.
@serve.deployment(ray_actor_options={"num_cpus": 0.1})
def f(*args):
    return "hello world! \n"

# Creates a Ray Serve deployment for a driver that calls the "hello world" function.
@serve.deployment(ray_actor_options={"num_cpus": 0.1})
class BasicDriver:
    def __init__(self, dag: RayServeDAGHandle):
        self.dag = dag

    # Asynchronously calls the "hello world" function.
    async def __call__(self): 
        return await (await self.dag.remote())

# Binds the "hello world" function and the driver to the same deployment.
FNode = f.bind()
DagNode = BasicDriver.bind(FNode)