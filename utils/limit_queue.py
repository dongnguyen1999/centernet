class LimitQueue:
    def __init__(self, limit=15):
        self.limit = limit
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)
        if len(self.queue) > self.limit:
            self.dequeue()
    
    def dequeue(self):
        self.queue.pop(0)

    def top(self):
        if len(self.queue) > 0:
            return self.queue[0]
        return None