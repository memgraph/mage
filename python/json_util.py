import mgp
import json
import urllib.request


@mgp.read_proc
def load_from_path(ctx: mgp.ProcCtx, path: str) -> mgp.Record(objects=mgp.List[object]):
    """
    Procedure to load JSON from a local file. 

    Parameters
    ----------
    path : str
        Path to the JSON that is being loaded.
    """
    with open(path) as json_file:
        objects = json.load(json_file)

        if type(objects) is dict:
            objects = [objects]

    return mgp.Record(objects=objects)


@mgp.read_proc
def load_from_url(ctx: mgp.ProcCtx, url: str) -> mgp.Record(objects=mgp.List[object]):
    """
    Procedure to load JSON from a remote address. 

    Parameters
    ----------
    path : str
        URL to the JSON that is being loaded.
    """
    with urllib.request.urlopen(url) as json_url:
        objects = json.loads(json_url.read().decode())

        if type(objects) is dict:
            objects = [objects]

    return mgp.Record(objects=objects)
