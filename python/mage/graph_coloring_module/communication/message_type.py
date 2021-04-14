from enum import Enum


class MessageType(Enum):
    """
    The message type STOP means that the solution is found.
    The process that received this message should terminate its execution.

    The message type FROM_NEXT_CHUNK means that the message comes from the
    process that contains the next part of the population. The process that
    received the message of this type should change its next individual to
    the individual that is sent in the message.

    The message type FROM_PREV_CHUNK means that the message comes from the
    process that contains the previous part of the population. The process
    that received the message of this type should change its previous
    individual to the individual that is sent in the message.
    """

    STOP = 0
    FROM_NEXT_CHUNK = 1
    FROM_PREV_CHUNK = -1
