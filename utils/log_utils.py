from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_lr(self, step, lr):
        self.writer.add_scalar('learning_rate', lr, step)

    def log_training(self, step, loss_metric):
        losses = loss_metric.get_avg_losses(flush=True)
        for key in losses.keys():
            self.writer.add_scalars(f'Loss/{key}', {'train': losses[key]}, step)
        total = loss_metric.calculate_loss(losses)
        self.writer.add_scalars('Loss/total', {'train': total}, step)

    def log_eval(self, step, loss_metric):
        losses = loss_metric.get_avg_losses(flush=True)
        for key in losses.keys():
            self.writer.add_scalars(f'Loss/{key}', {'val': losses[key]}, step)
        total = loss_metric.calculate_loss(losses)
        self.writer.add_scalars('Loss/total', {'val': total}, step)

    def log_ap(self, step, ap, dataset_name):
        print(f'{dataset_name}:\t AP:\t {ap}')
        self.writer.add_scalar(f'test_ap/{dataset_name}', ap, step)

    def log_probs(self, step, probs):
        self.writer.add_scalars('probs', {'p': probs[0]}, step)
        self.writer.add_scalars('probs', {'n': probs[1]}, step)
