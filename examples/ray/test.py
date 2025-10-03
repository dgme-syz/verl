import warnings
import os
from typing import Optional

import ray
import socket
from vllm import LLM
warnings.filterwarnings("ignore")

from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, merge_resource_pool
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, Execute, register

os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,6"
os.environ["RAY_DEDUP_LOGS"] = "0"
model_path = os.path.join(os.environ.get("MODEL", None), "Qwen2.5-7B-Instruct") 

def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    candidates = (29500, 23456, 12355, 12345)
    for p in candidates:
        if _is_port_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ensure_master_addr_port(addr: Optional[str] = None, port: Optional[int] = None) -> None:
    """
    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

    - Respects existing environment variables.
    - Defaults `MASTER_ADDR` to localhost if unset.
    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
    if port is None and env_port not in {"", "0", "auto"}:
        try:
            port = int(env_port)
        except ValueError:
            pass

    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)


@ray.remote
class Tasker(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(Dispatch.ONE_TO_ALL)
    def f(self, tp_size: int = 4):

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            # ensure tp groups to ensure sampling results are the same across workers
            seed=self.rank // tp_size, 
            distributed_executor_backend=None
        )
    
    @register(Dispatch.ONE_TO_ALL)
    def g(self, prompts: list[str]):
        outs = self.llm.generate(prompts=prompts)
        resps = []
        for out in outs:
            for id in range(len(out.outputs)):
                resp_ids = out.outputs[id].token_ids
                resps.append(resp_ids)
        return resps 
    
ray.init()
resource_pool = RayResourcePool([4], use_gpu=True)

class_with_args = RayClassWithInitArgs(cls=Tasker)
print("[Done] cls")
worker_group = RayWorkerGroup(resource_pool, class_with_args)
worker_group.f(tp_size=1)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
resp_ids = worker_group.g([
    "1+1=", 
    "2+2=",
    "2+1=",
    "2+5=",
])
print(resp_ids)
print(resp_ids[0] is resp_ids[1])
print(tokenizer.batch_decode(resp_ids[0], skip_special_tokens=True))
