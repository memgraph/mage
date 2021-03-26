from enum import Enum


class MessageType(Enum):
    STOP = 0
    FROM_NEXT_CHUNK = 1
    FROM_PREV_CHUNK = -1
