import mgp
import subprocess
import sys
import os

# os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1" # NOTE: With this HB can't download the model...
# os.environ["PYTHONNOUSERSITE"] = "1"
# sys.path.append("/home/memgraph/.local/lib/python3.12/site-packages")

# NOTE: Dirty fix for the issue of failed HB logger...
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
try:
    from huggingface_hub.utils import _logging
except ImportError:
    # Fallback if _logging is unavailable
    import logging
    _logging = logging

EXCLUDE_PROPERTIES = {"embedding"}
BATCH_SIZE = 2000
# Memory usage examples:
#  * 1k batch on https://github.com/datacharmer/test_db uses 4.6GB vRAM peak.
logger: mgp.Logger = mgp.Logger()

# TODO(gitbuda): Parametrize the number of devices used.
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # NOTE: This seems to no be working...
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"---- DEVICE: {device}")


# TODO(gitbuda): The input should be a list of vertices because we could select + batch.
@mgp.write_proc
def compute_embeddings(ctx: mgp.ProcCtx) -> mgp.Record(success=bool):
    logger.info(
        f"compute_embeddings: starting (device={device}, py_exec={sys.executable}, py_ver={sys.version.split()[0]})"
    )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.error(f"sentence-transformers not available: {e}")
        return mgp.Record(success=False)
    try: 
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    except Exception as e:
        logger.warning(f"Failed to load model: {e}")
        return mgp.Record(success=False)

    try:
        batch_vertices = []
        batch_data = []
        for vertex in ctx.graph.vertices:
            # TODO: parametrize the excluded properties
            node_data = " ".join(label.name for label in vertex.labels) + " " + " ".join(
                f"{key}: {value}"
                for key, value in vertex.properties.items()
                if key not in EXCLUDE_PROPERTIES
            )
            batch_vertices.append(vertex)
            batch_data.append(node_data)
            # Process batch when it reaches BATCH_SIZE
            if len(batch_vertices) == BATCH_SIZE:
                # Compute embeddings for the batch
                # TODO(gitbuda): Use pytorch data loader with prefetching.
                batch_embeddings = model.encode(
                    batch_data,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                # Update vertex properties with computed embeddings
                for vertex, embedding in zip(batch_vertices, batch_embeddings):
                    # TODO: parametrize the property name
                    vertex.properties["embedding"] = embedding.tolist()
                # Clear the batch lists for next iteration
                batch_vertices.clear()
                batch_data.clear()

        # Process remaining vertices in the last batch
        if batch_vertices:
            batch_embeddings = model.encode(
                batch_data,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            for vertex, embedding in zip(batch_vertices, batch_embeddings):
                # TODO: parametrize the property name
                vertex.properties["embedding"] = embedding.tolist()

        return mgp.Record(success=True)
    except Exception as e:
        # Handle exceptions by returning failure status
        logger.error(f"Failed to compute embedding for node: {e}")
        return mgp.Record(success=False)