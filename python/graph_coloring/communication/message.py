from telco.components.individual import Individual


class Message:
    """A class that represents the message. Messages are exchanged between processes.
    The message contains an individual representing a single coloring of the graph. The message
    has a type, -1 or 1. The type is 1 if the message was sent from the process containing
    the next chunk of the population and -1 if it was sent from the process containing the
    preceding one population chunk."""

    def __init__(
            self,
            data: Individual,
            msg_type: int):
        self._data = data
        self._msg_type = msg_type

    @property
    def data(self) -> Individual:
        """Returns the individual contained in the message."""
        return self._data

    @property
    def msg_type(self) -> int:
        """Returns the message type. The message type is 1 if the message was sent from the
        next chunkof the population and -1 if it was sent from the preceding one."""
        return self._msg_type
