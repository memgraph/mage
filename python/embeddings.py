import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import mgp

sys.path.append(os.path.join(os.path.dirname(__file__), "embed_worker"))

logger: mgp.Logger = mgp.Logger()

EXCLUDE_PROPERTIES = {"embedding"}
BATCH_SIZE = 2000
MODEL_ID = "all-MiniLM-L6-v2"


def build_texts(vertices):
    out = []
    for v in vertices:
        txt = " ".join(lbl.name for lbl in v.labels) + " " + " ".join(
            f"{k}: {val}" for k, val in v.properties.items() if k not in EXCLUDE_PROPERTIES
        )
        out.append(txt)
    return out


def split_slices(n_items: int, n_parts: int):
    base, rem = divmod(n_items, n_parts)
    start = 0
    slices = []
    for i in range(n_parts):
        end = start + base + (1 if i < rem else 0)
        slices.append((start, end))
        start = end
    return slices


def get_visible_gpus():
    # Avoid creating a CUDA context in the parent if possible
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        return [int(x) for x in out.strip().splitlines() if x.strip()]
    except Exception:
        try:
            import torch
            return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        except Exception:
            return []


@mgp.write_proc
def compute_embeddings(ctx: mgp.ProcCtx) -> mgp.Record(success=bool):
    logger.info(
        f"compute_embeddings: starting (py_exec={sys.executable}, py_ver={sys.version.split()[0]})"
    )
    try:
        # Parent imports are okay; workers import only embed_worker
        import embed_worker  # <-- our pure worker module
    except Exception as e:
        logger.error(f"Failed to import worker module: {e}")
        return mgp.Record(success=False)

    try:
        vertices = list(ctx.graph.vertices)
        texts = build_texts(vertices)
        n = len(texts)
        if n == 0:
            logger.info("No vertices to process.")
            return mgp.Record(success=True)

        gpus = get_visible_gpus()
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")

        # CPU fallback (single process in parent)
        if not gpus:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_ID, device="cpu")
                embs = model.encode(
                    texts,
                    batch_size=min(BATCH_SIZE, n),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for v, e in zip(vertices, embs.tolist()):
                    v.properties["embedding"] = e
                logger.info(f"Processed {n} vertices on CPU.")
                return mgp.Record(success=True)
            except Exception as e:
                logger.error(f"CPU path failed: {e}")
                return mgp.Record(success=False)

        # Multi-GPU via spawn
        slices = split_slices(n, len(gpus))
        tasks = []
        for gpu, (a, b) in zip(gpus, slices):
            if a < b:
                tasks.append((gpu, MODEL_ID, texts[a:b], BATCH_SIZE, a, b))

        results = []
        total = 0

        mp.set_executable("/usr/bin/python3")
        ctx_spawn = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx_spawn) as ex:
            fut2info = {
                ex.submit(embed_worker.encode_chunk, t[0], t[1], t[2], t[3]): (t[0], t[4], t[5])
                for t in tasks
            }
            for fut in as_completed(fut2info):
                gpu, a, b = fut2info[fut]
                try:
                    count, embs = fut.result()
                    if count != (b - a) or len(embs) != (b - a):
                        logger.error(f"GPU {gpu} returned mismatched count {count} for slice [{a}:{b})")
                        continue
                    results.append((a, b, embs))
                    total += count
                    logger.info(f"GPU {gpu} returned {count} embeddings for slice [{a}:{b}).")
                except Exception as e:
                    logger.error(f"Worker on GPU {gpu} failed: {e}")

        # Write back
        for a, b, embs in results:
            for i, e in enumerate(embs, start=a):
                vertices[i].properties["embedding"] = e

        logger.info(f"Successfully processed {total}/{n} vertices across {len(gpus)} GPU(s).")
        return mgp.Record(success=(total == n))

    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        return mgp.Record(success=False)
