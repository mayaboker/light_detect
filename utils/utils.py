import yaml
from jinja2 import Environment, BaseLoader

def load_yaml(yaml_path: str):
    with open(yaml_path, 'r') as y:
        jinja_str = Environment(loader=BaseLoader()).from_string(y.read()).render()
        return yaml.safe_load(jinja_str)