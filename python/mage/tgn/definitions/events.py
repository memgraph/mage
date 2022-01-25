class Event:
    def __init__(self, source: int, timestamp: int):
        super(Event, self).__init__()
        self.source = source
        self.timestamp = timestamp

    def __str__(self):
        return "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)


class NodeEvent(Event):
    def __init__(self, source: int, timestamp: int):
        super(NodeEvent, self).__init__(source, timestamp)

    def __str__(self):
        return "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)


class InteractionEvent(Event):
    def __init__(self, source: int, dest: int, timestamp: int, edge_indx:int):
        super(InteractionEvent, self).__init__(source, timestamp)
        self.dest = dest
        self.edge_indx=edge_indx

    def __str__(self):
        return "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)
