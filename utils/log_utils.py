from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_lr(self, step, lr):
        self.writer.add_scalar('learning_rate', lr, step)

    def log_training(self, step, loss_metric):
        pass

    def log_probs(self, step, probs):
        self.writer.add_scalars('probs', {'p': probs[0]}, step)
        self.writer.add_scalars('probs', {'n': probs[1]}, step)
