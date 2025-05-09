import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.total_duration = 0
        self.num_tokens = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        duration = time.time() - self.start_time
        self.total_duration += duration
        self.start_time = time.time()
        return duration
    
    def get_total_duration(self):
        return self.total_duration
    
    def get_average_duration(self):
        if self.num_tokens == 0:
            return 0
        return self.total_duration / self.num_tokens
    
