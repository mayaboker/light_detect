from train import Trainer
from utils.utils import load_yaml


cfg = load_yaml('config.yaml')
Trainer(cfg)

