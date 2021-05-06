from torch.utils.tensorboard import SummaryWriter
import numpy as np

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

    def log_pr_curve(self, step, pr):
        fars = np.arange(0.001, 0.1, 0.001)        
        pt = np.zeros((fars.shape[0], 2))
        fars_step = 0            
        thresh_num = pr.shape[0]
        for i in range(pr.shape[0]):
            self.writer.add_scalars(f'PR', {str(step): pr[i, 0] * 1000}, pr[i, 1]* 1000)
            if fars_step < fars.shape[0] and pr[i, 0] < 1-fars[fars_step]:
                pt[fars_step] = (1- (i+1) / thresh_num, pr[i, 0])
                fars_step += 1
        self.log_pt_curve(step, pt)
    
    def log_pt_curve(self, step, pt):
        for i in range(pt.shape[0]):
            self.writer.add_scalars(f'PT', {str(step): pt[i, 1]}, pt[i, 0] * 1000)
