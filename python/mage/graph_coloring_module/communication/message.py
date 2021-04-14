from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.communication.message_type import MessageType


class Message:
    def __init__(self, data: Individual, msg_type: MessageType, process_id: int):
        self._data = data
        self._msg_type = msg_type
        self._process_id = process_id

    @property
    def data(self) -> Individual:
        """Returns the individual that is sent in the message."""
        return self._data

    @property
    def msg_type(self) -> MessageType:
        """Returns the message type."""
        return self._msg_type

    @property
    def process_id(self) -> int:
        """Returns the identifier of the process that sent the message."""
        return self._process_id
