import mgp
import json
import kafka


@mgp.read_proc
def send_json(
    topic: str,
    host: str = '127.0.0.1',
    port: str = '9092',
    **kwargs: any,
) -> mgp.Record(str):
    """Pushes named data to a kafka stream.

    Example:
        MATCH (a)
        WITH collect(a) AS nodes
        CALL kafka_stream.send_json('nodes', nodes=nodes) YIELD message
        RETURN message

    :param str topic: Kafka stream topic name.
    :param str host: Host IP of the Kafka stream (default 127.0.0.1).
    :param str port: Port of the Kafka stream (default 9092).
    :return str message: String message sent to Kafka.
    """
    message = json.dumps(kwargs)
    producer = kafka.KafkaProducer(bootstrap_servers=[f'{host}:{port}'])
    producer.send(topic, message.encode('utf-8'))
    producer.flush()
    return mgp.Record(message=message)
