import torch

from composer.core import Callback, Logger, State
from composer.models.loss import soft_cross_entropy


class LossMonitor(Callback):

    def __init__(
        self,
        epochs,
        num_batches,
        batch_size,
    ):
        super().__init__()
        self.loss_values = torch.zeros(1, num_batches, batch_size, 512, 512)
        self.epochs = [epochs]
        self.num_batches = num_batches
        self.batch_size = batch_size

    # def init(self, state: State, logger: Logger) -> None:

    #     def loss(outputs, batch):
    #         target = batch[1]
    #         loss = soft_cross_entropy(outputs, target, ignore_index=-1, reduction='none')
    #          return loss

    #     state.model.loss = loss

    def before_loss(self, state: State, logger: Logger) -> None:
        if (state.timer._epoch in self.epochs) and (state.timer._batch_in_epoch < self.num_batches):
            with torch.no_grad():
                outputs = state.model.forward(state.batch)
                loss = torch.nn.functional.cross_entropy(outputs, state.batch[1], ignore_index=-1, reduction='none')
                print(loss.shape)
            #self.state.loss = self.state.loss.mean()
