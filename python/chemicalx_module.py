import io
import json
import os
import urllib.request
from typing import Dict

import mgp
import numpy as np
import pandas as pd
import torch
from chemicalx import pipeline
from chemicalx.models import DeepSynergy
from chemicalx.data import DrugCombDB
from tabulate import tabulate


def load_raw_json_data(path: str) -> Dict:
    with open(path, 'r') as fh:
        text = fh.read()
    raw_data = json.loads(text)
    return raw_data


def get_drug_features() -> Dict:
    path = "/home/antonio/work/memgraph/github-repos/datasets/two-sides/drug_set.json"
    raw_data = load_raw_json_data(path)
    raw_data = {
        key: {"smiles": value["smiles"], "features": np.array(value["features"])}
        for key, value in raw_data.items()
    }
    return raw_data


def get_context_features() -> Dict:
    """Get the context feature set."""
    path = "/home/antonio/work/memgraph/github-repos/datasets/two-sides/context_set.json"
    raw_data = load_raw_json_data(path)
    raw_data = {k: np.array(v) for k, v in raw_data.items()}
    return raw_data

def load_raw_csv_data() -> pd.DataFrame:
    """Load a CSV dataset at the given path.

    :param path: The path to the triples CSV file.
    :returns: A pandas DataFrame with the data.
    """
    path = "/home/antonio/work/memgraph/github-repos/datasets/two-sides/labeled_triples.csv"
    with open(path) as fh:
        types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        raw_data = pd.read_csv(fh, encoding="utf8", sep=",", dtype=types)
    return raw_data

def create_cypherl(save_file_path:str):
    logger = mgp.Logger()
    context_features = get_context_features()
    drug_features = get_drug_features()
    lines = []

    lines.append('CREATE INDEX ON :Context(id);')
    lines.append('CREATE INDEX ON :Drug(id);')

    # We don't need to store this inside Memgraph. Because we will only store encoding.
    # Once we get set of contexts from edge property 'context', we can make one hot encoding of
    # every 'context' property and use that from then onwards.
    # for k, v in context_features.items():
    #     out = f"MERGE (n:Context {{id:'{k}', encoding:{np.array(v).tolist()}}});"
    #     logger.critical(out)
    #     lines.append(out)

    for k, v in drug_features.items():
        out = f"MERGE (n:Drug {{id:'{k}', smiles:'{v['smiles']}', features:{np.array(v['features']).tolist()} }});"
        #logger.critical(out)
        lines.append(out)

    pd_df = load_raw_csv_data()

    for index, row in pd_df.iterrows():
        out = f"MERGE (n:Drug {{id:'{row['drug_1']}'}}) " \
              f"MERGE (n1:Drug {{id:'{row['drug_2']}'}}) " \
              f"CREATE (n)-[:CONNECTED {{context:'{row['context']}', label:{row['label']}}}]->(n1);"
        lines.append(out)
    with open(save_file_path, 'a') as fh:
         fh.write("\n".join(lines))


@mgp.read_proc
def run(ctx: mgp.ProcCtx) -> mgp.Record():
    create_cypherl("/home/antonio/work/memgraph/github-repos/datasets/two-sides/twosides.cypherl")

    raw_drug_data = {}

    raw_dict_data = {'drug_1': [], 'drug_2': [], 'context':[], 'label':[] }

    for vertex in ctx.graph.vertices:
        raw_drug_data[vertex.properties.get('id')] = {
            'smiles': vertex.properties.get('smiles'),
            'features': np.array(vertex.properties.get('features')).reshape(1, -1)
        }

        for edge in vertex.out_edges:
            context = edge.properties.get('context')
            label = edge.properties.get('label')
            from_vertex = edge.from_vertex
            to_vertex = edge.to_vertex

            raw_dict_data['drug_1'].append(from_vertex.properties.get('id'))
            raw_dict_data['drug_2'].append(to_vertex.properties.get('id'))
            raw_dict_data['context'].append(context)
            raw_dict_data['label'].append(label)

    all_contexts = set(raw_dict_data['context'])

    #this we use here
    pd_df = pd.DataFrame.from_dict(raw_dict_data)


    # model = DeepSynergy(context_channels=112, drug_channels=256)
    # dataset = DrugCombDB()
    #
    # context_features = dataset.get_context_features()
    # drug_features = dataset.get_drug_features()
    # labeled_triples = dataset.get_labeled_triples()
    #
    #
    # results = pipeline(
    #     dataset=dataset,
    #     model=model,
    #     # Data arguments
    #     batch_size=5120,
    #     context_features=True,
    #     drug_features=True,
    #     drug_molecules=False,
    #     # Training arguments
    #     epochs=100,
    # )


    # Outputs information about the AUC-ROC, etc. to the console.

    # logger.info(tabulate(sorted(results.metrics.items()), headers=["Metric", "Value"]))

    # Save the model, losses, evaluation, and other metadata.
    # results.save("~/test_results/")
    return mgp.Record()
