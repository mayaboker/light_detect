import yaml
from jinja2 import Environment, BaseLoader

def load_yaml(yaml_path: str):
    with open(yaml_path, 'r') as y:
        jinja_str = Environment(loader=BaseLoader()).from_string(y.read()).render()
        return yaml.safe_load(jinja_str)


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

class ProbsAverageMeter:
    def __init__(self):
        self.pos_meter = AverageMeter()
        self.neg_meter = AverageMeter()
        self.reset()
    
    def reset(self):
        self.pos_meter.reset()
        self.neg_meter.reset()

    def update(self, probs, n=1):
        self.pos_meter.update(probs[0], n)
        self.neg_meter.update(probs[1], n)

    def get_average(self):
        return self.pos_meter.avg, self.neg_meter.avg