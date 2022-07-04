from typing import Union
from functools import lru_cache

import mgp
import numpy as np
import torch
from chemicalx.models import DeepSynergy
from chemicalx.data import (
    BatchGenerator,
    DatasetLoader,
    DrugFeatureSet,
    ContextFeatureSet,
    LabeledTriples,
)
from sklearn.preprocessing import LabelBinarizer
from chemicalx.models import Model

model: Model = None
context_set: ContextFeatureSet = None

class MemgraphDatasetLoader(DatasetLoader):
    """A dataset loader that processes and caches data locally."""

    DRUG_1 = "drug_1"
    DRUG_2 = "drug_2"
    ID = "id"
    SMILES = "smiles"
    FEATURES = "features"
    CONTEXT = "context"
    LABEL = "label"

    def __init__(self, ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge] = None):
        """Instantiate the local dataset loader."""
        self.drug_data = {}
        self.interaction_data = {
            self.DRUG_1: [],
            self.DRUG_2: [],
            self.CONTEXT: [],
            self.LABEL: [],
        }

        for vertex in ctx.graph.vertices:
            self.drug_data[str(vertex.properties.get(self.ID))] = {
                self.SMILES: vertex.properties.get(self.SMILES),
                self.FEATURES: np.array(vertex.properties.get(self.FEATURES)).reshape(
                    1, -1
                ),
            }

        for edge in self.edge_generator(edges if edges else ctx):
            context = edge.properties.get(self.CONTEXT)
            label = edge.properties.get(self.LABEL)
            from_vertex = edge.from_vertex
            to_vertex = edge.to_vertex

            self.interaction_data[self.DRUG_1].append(
                str(from_vertex.properties.get(self.ID))
            )
            self.interaction_data[self.DRUG_2].append(
                str(to_vertex.properties.get(self.ID))
            )
            self.interaction_data[self.CONTEXT].append(context)
            self.interaction_data[self.LABEL].append(label)

        context = list(set(self.interaction_data[self.CONTEXT]))
        self.context_data = {
            label: encoding
            for label, encoding in zip(context, LabelBinarizer().fit_transform(context))
        }
    
    @staticmethod
    def edge_generator(source: Union[mgp.ProcCtx, mgp.List[mgp.Edge]]):
        if type(source) == mgp.ProcCtx:
            for vertex in source.graph.vertices:
                for edge in vertex.out_edges:
                    yield edge
        else:
            for edge in source:
                yield edge

    @lru_cache(maxsize=1)
    def get_drug_features(self) -> DrugFeatureSet:
        """Get the drug feature set."""
        return DrugFeatureSet.from_dict(self.drug_data)

    @lru_cache(maxsize=1)
    def get_context_features(self) -> ContextFeatureSet:
        """Get the context feature set."""
        return ContextFeatureSet.from_dict(self.context_data)

    @lru_cache(maxsize=1)
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples dataframe."""
        return LabeledTriples(self.interaction_data)


@mgp.read_proc
def train(
    ctx: mgp.ProcCtx,
    edges: mgp.Nullable[mgp.List[mgp.Edge]] = None,
    learning_rate: float = 1e-4,
    batch_size: int = 1024,
    epochs: int = 100,
) -> mgp.Record():
    global model, context_set
    model = DeepSynergy(context_channels=112, drug_channels=256)
    model.train()

    loader = MemgraphDatasetLoader(ctx, edges)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    context_set = loader.get_context_features()
    drug_set = loader.get_drug_features()
    triples = loader.get_labeled_triples()
    generator = BatchGenerator(
        batch_size=batch_size,
        context_features=True,
        drug_features=True,
        drug_molecules=False,
        context_feature_set=context_set,
        drug_feature_set=drug_set,
        labeled_triples=triples,
    )

    for epoch in range(epochs):
        for batch in generator:
            optimizer.zero_grad()
            prediction = model(
                batch.context_features,
                batch.drug_features_left,
                batch.drug_features_right,
            )
            loss_value = loss(prediction, batch.labels)
            loss_value.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} loss: {float(loss_value)}")

    return mgp.Record()


@mgp.read_proc
def predict(
    ctx: mgp.ProcCtx,
    edges: mgp.Nullable[mgp.List[mgp.Edge]] = None,
    batch_size: int = 32,
) -> mgp.Record(predictions=list):
    global model
    if model is None or context_set is None:
        raise RuntimeError("The model is not initialized!")
    model.eval()

    loader = MemgraphDatasetLoader(ctx, edges)
    drug_set = loader.get_drug_features()
    triples = loader.get_labeled_triples()
    generator = BatchGenerator(
        batch_size=batch_size,
        context_features=True,
        drug_features=True,
        drug_molecules=False,
        context_feature_set=context_set,
        drug_feature_set=drug_set,
        labeled_triples=triples,
    )
    predictions = []
    for batch in generator:
        prediction_batch = model(
            batch.context_features, batch.drug_features_left, batch.drug_features_right
        )
        prediction_batch = prediction_batch.detach().cpu().numpy()
        for prediction in prediction_batch:
            predictions.append(float(prediction))

    return mgp.Record(predictions=predictions)
