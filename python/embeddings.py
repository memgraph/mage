import mgp
from sentence_transformers import SentenceTransformer

# Initialize default model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Exclude certain properties from embedding computation
EXCLUDE_PROPERTIES = {"embedding"}

@mgp.read_proc
def compute_embeddings(ctx: mgp.ProcCtx) -> mgp.Record(node_id=int, success=bool):
    try:
        # Fetch all nodes from the graph
        for vertex in ctx.graph.vertices:
            # Combine node labels and properties into a single string, excluding specified properties
            node_data = " ".join(vertex.labels) + " " + " ".join(
                f"{key}: {value}"
                for key, value in vertex.properties.items()
                if key not in EXCLUDE_PROPERTIES
            )

            # Compute the embedding
            node_embedding = model.encode(node_data).tolist()

            # Update the node with the computed embedding
            vertex.properties["embedding"] = node_embedding

        return mgp.Record(node_id=vertex.id, success=True)

    except Exception as e:
        # Handle exceptions by returning failure status
        return mgp.Record(node_id=None, success=False)
