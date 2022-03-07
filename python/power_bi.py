import mgp
import requests
import json
from datetime import datetime


API_URLS = {}
ORDER_NUMBER = 0


@mgp.read_proc
def create_push_stream(
    context: mgp.ProcCtx,
    stream_name: str,
    api_url: str,
) -> mgp.Record():
    global API_URLS

    API_URLS[stream_name] = api_url

    return mgp.Record()


@mgp.read_proc
def show_streams(
    context: mgp.ProcCtx,
) -> mgp.Record(name=str, api_url=str):
    records = []

    for k, v in API_URLS.items():
        records.append(mgp.Record(name=k, api_url=v))

    return records


@mgp.read_proc
def push(
    context: mgp.ProcCtx,
    stream_name: str,
    payload: mgp.Map,
) -> mgp.Record(status=str):

    if stream_name not in API_URLS:
        raise Exception("Power BI stream not defined!")

    api_url = API_URLS[stream_name]

    message = ""
    if isinstance(payload, dict):
        message = payload
    elif isinstance(payload, mgp.Vertex) or isinstance(payload, mgp.Edge):
        message = {x.name: x.value for x in payload.properties.items()}
    else:
        raise Exception("Can't have message type other than Map / Vertex / Edge")

    for k, v in message.items():
        if isinstance(v, datetime):
            message[k] = datetime.strftime(v, "%Y-%m-%dT%H:%M:%S")

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.request(
            method="POST", url=api_url, headers=headers, data=json.dumps(message)
        )
    except Exception as e:
        raise Exception(f"Error happened while sending results! {e}")

    return mgp.Record(status=str(response.status))
