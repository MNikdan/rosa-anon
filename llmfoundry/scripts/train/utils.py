from typing import List, Optional
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def get_checkpoint_and_refs_dir(
    model_id: str,
    bucket_uri: str,
    s3_sync_args: Optional[List[str]] = None,
    mkdir: bool = False,
) -> str:

    from transformers.utils.hub import TRANSFORMERS_CACHE

    f_hash = get_hash_from_bucket(bucket_uri, s3_sync_args)

    path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")

    refs_dir = os.path.join(path, "refs")
    checkpoint_dir = os.path.join(path, "snapshots", f_hash)

    if mkdir:
        os.makedirs(refs_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir, refs_dir


def get_download_path(model_id: str):
    from transformers.utils.hub import TRANSFORMERS_CACHE

    path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
    return path


def get_mirror_link(model_id: str) -> str:
    return f"s3://llama-2-weights/models--{model_id.replace('/', '--')}"


def swap_module(network, module_name, new_module):
    name_parts = module_name.split('.')
    parent = network
    for part in name_parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    last_part = name_parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)
