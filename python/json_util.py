from io import TextIOWrapper
import json
import mgp
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError


def extract_objects(file: TextIOWrapper):
    objects = json.load(file)
    if type(objects) is dict:
        objects = [objects]
    return objects


@mgp.read_proc
def load_from_path(ctx: mgp.ProcCtx, path: str) -> mgp.Record(objects=mgp.List[object]):
    """
    Procedure to load JSON from a local file.

    Parameters
    ----------
    path : str
        Path to the JSON that is being loaded.
    """
    file = Path(path)
    if file.exists():
        opened_file = open(file)
        objects = extract_objects(opened_file)
    else:
        raise FileNotFoundError("There is no file " + path)

    opened_file.close()

    return mgp.Record(objects=objects)


@mgp.read_proc
def load_from_url(ctx: mgp.ProcCtx, url: str) -> mgp.Record(objects=mgp.List[object]):
    """
    Procedure to load JSON from a remote address.

    Parameters
    ----------
    url : str
        URL to the JSON that is being loaded.
    """
    request = Request(url)
    request.add_header("User-Agent", "MAGE module")
    try:
        content = urlopen(request)
    except URLError as url_error:
        raise url_error
    else:
        objects = extract_objects(content)

    return mgp.Record(objects=objects)
