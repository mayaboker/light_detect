class LossMetric:
    def __init__(self, cfg):
        loss_weights = cfg['train']['loss_weights']
        self.losses = {}
        self.weights = {}

        for key, weight in loss_weights.items():
            self.losses[key] = 0
            self.weights[key] = weight
        
        self.step = 0

    def add_sample(self, loss_dict):
        for key in loss_dict.keys():
            self.losses[key] += loss_dict[key].item()
        self.step += 1

    def get_avg_losses(self, flush=True):
        ret_losses = {}
        for key in self.losses.keys():
            ret_losses[key] = self.losses[key] / self.step
            if flush:
                self.losses[key] = 0
        if flush:
            self.step = 0
        return ret_losses

    def calculate_loss(self, loss_dict):
        loss = 0
        for key in loss_dict.keys():
            loss += loss_dict[key] * self.weights[key]
        return loss

    def get_total(self):
        loss = 0
        for key in self.losses.keys():
            loss += self.losses[key] * self.weights[key]
        return loss