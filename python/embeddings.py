import mgp
import subprocess
import sys


EXCLUDE_PROPERTIES = {"embedding"}

logger: mgp.Logger = mgp.Logger()

@mgp.write_proc
def compute_embeddings(ctx: mgp.ProcCtx, node: mgp.Vertex) -> mgp.Record(embedding_string=str, success=bool):

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Make sure pip is there
        try:
            subprocess.check_call([sys.executable, "ensurepip"])
        except subprocess.CalledProcessError:
            logger.error("Failed to ensure pip is available")
            return mgp.Record(embedding_string="", success=False)
        
        # Install the sentence-transformers package
        try:
            subprocess.check_call([
                sys.executable, "pip", "install", "sentence-transformers"
            ])
            from sentence_transformers import SentenceTransformer
        except subprocess.CalledProcessError:
            logger.error("Failed to install the sentence-transformers package")
            return mgp.Record(embedding_string="", success=False)

    try: 
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return mgp.Record(embedding_string="", success=False)

    try:

        for vertex in ctx.graph.vertices:

            # Test id: 555 name: Pero last_name: Peric nums: (1, 2, 3) birthday: 1947-07-30 maps: {'day': 30, 'month': 7, 'year': 1947} lap: 0:02:02.000033

            #TODO: parametrize the exluded properties
            node_data = " ".join(label.name for label in vertex.labels) + " " + " ".join(
                f"{key}: {value}"
                for key, value in vertex.properties.items()
                if key not in EXCLUDE_PROPERTIES
            )
            # Compute the embedding
            node_embedding = model.encode(node_data)

            #TODO: parametrize the property name
            vertex.properties["embedding"] = node_embedding.tolist()

        return mgp.Record(embedding_string=node_data, success=True)

    except Exception as e:
        # Handle exceptions by returning failure status
        logger.error(f"Failed to compute embedding for node: {e}")
        return mgp.Record(embedding_string="", success=False)
