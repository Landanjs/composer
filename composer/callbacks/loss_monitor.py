import torch

from composer.core import Callback, State
from composer.loggers import Logger
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

    def init(self, state: State, logger: Logger) -> None:
        loss_fn = state.model.loss

        def unreduced_loss(outputs, batch):
            return loss_fn(outputs, batch, reduction='none')

        state.model.loss = unreduced_loss

    def after_loss(self, state: State, logger: Logger):
        logger.log_metric(state.loss)

    def before_backward(self, state: State, logger: Logger):
        _, targets = state.batch
        state.loss = state.loss[targets == -1].mean()

    def before_loss(self, state: State, logger: Logger) -> None:
        if (state.timer._epoch in self.epochs) and (state.timer._batch_in_epoch < self.num_batches):
            with torch.no_grad():
                outputs = state.model.forward(state.batch)
                loss = torch.nn.functional.cross_entropy(outputs, state.batch[1], ignore_index=-1, reduction='none')
                print(loss.shape)
            #self.state.loss = self.state.loss.mean()
