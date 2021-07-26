import mgp
import json
import kafka


@mgp.read_proc
def send_json(
    context: mgp.ProcCtx,
    topic: str,
    host: str = 'localhost',
    port: str = '9092',
    **kwargs: any,
) -> mgp.Record():
    """Pushes named data to a kafka stream."""
    message = json.dumps(kwargs)
    producer = kafka.KafkaProducer(bootstrap_servers=[f'{host}:{port}'])
    producer.send(topic, message.encode('utf-8'))
    producer.flush()
    return mgp.Record(message=message)
