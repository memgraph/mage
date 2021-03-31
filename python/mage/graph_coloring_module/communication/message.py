from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.communication.message_type import MessageType


class Message:
    def __init__(self, data: Individual, msg_type: MessageType, proc_id: int):
        self._data = data
        self._msg_type = msg_type
        self._proc_id = proc_id

    @property
    def data(self) -> Individual:
        """Returns the individual contained in the message."""
        return self._data

    @property
    def msg_type(self) -> MessageType:
        """Returns the message type. The message type is 1 if the message was sent from the
        next chunkof the population and -1 if it was sent from the preceding one."""
        return self._msg_type

    @property
    def proc_id(self) -> int:
        return self._proc_id
