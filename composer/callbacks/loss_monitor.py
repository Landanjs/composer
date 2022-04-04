import torch

import wandb
from composer.core import Callback, State
from composer.loggers import Logger


class LossMonitor(Callback):

    def __init__(
        self,
        epoch_interval,
        num_batches,
        batch_size,
    ):
        super().__init__()
        #self.loss_values = torch.zeros(1, num_batches, batch_size, 512, 512)
        self.epoch_interval = epoch_interval
        self.num_batches = num_batches
        #self.batch_size = batch_size

    def init(self, state: State, logger: Logger) -> None:
        loss_fn = state.model.loss

        def unreduced_loss(outputs, batch):
            return loss_fn(outputs, batch, reduction='none')

        state.model.loss = unreduced_loss

    def after_loss(self, state: State, logger: Logger):
        if ((int(state.timer.epoch) % self.epoch_interval) == 0) and (state.timer._batch_in_epoch < self.num_batches):
            #data = [[loss.item()] for loss in state.loss.detach().cpu().flatten()]
            #table = wandb.Table(data=data, columns=["losses"])
            _, targets = state.batch
            logger.data_batch({"loss/unreduced": wandb.Histogram(state.loss[targets != -1].detach().cpu())})
            #logger.data_batch(
            #    {"loss/unreduced": wandb.plot.histogram(table, "losses", title="Pixel-loss distribution")})
        with state.precision_context:
            _, targets = state.batch
            loss = state.loss[targets != -1]
            #print(loss.shape)
            inds = torch.argsort(loss.detach(), descending=True)
            state.loss = loss[inds[:int(len(inds) * 0.9)]].mean()

            #state.loss = state.loss[targets != -1].mean()
