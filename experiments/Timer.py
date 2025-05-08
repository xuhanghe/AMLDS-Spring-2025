import time

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.durations = []

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.durations.append(time.time() - self.start_time)
        self.start_time = time.time()
    
    def get_average_duration(self):
        if not self.durations:
            return 0
        return sum(self.durations) / len(self.durations)
    
