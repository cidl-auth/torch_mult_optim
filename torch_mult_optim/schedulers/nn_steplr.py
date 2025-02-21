import logging
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class NNStepLR():
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self,
                 optimizer,
                 step_size,
                 gamma_in=1.0,
                 gamma_out=1.0,
                 last_epoch=-1,
                 verbose=False):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma_in = gamma_in
        self.gamma_out = gamma_out
        self.last_epoch = last_epoch
        # super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self) -> dict:

        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ('optimizer', 'gamma_in', 'gamma_out', 'step_size')
        }

        state_dict['gamma_in'] = self.gamma_in
        state_dict['gamma_out'] = self.gamma_out
        state_dict['step_size'] = self.step_size

        return state_dict

    def load_state_dict(self, state_dict):
        pass

    def step(self) -> None:
        self.last_epoch += 1
        if (self.last_epoch != 0) and (self.last_epoch % self.step_size == 0):
            for i, param_group in enumerate(self.optimizer.param_groups):
                if 'lr_in' in param_group:
                    param_group['lr_in'] = param_group['lr_in'] * self.gamma_in
                    logging.debug('Updated lr_in=', param_group['lr_in'])
                if 'lr_out' in param_group:
                    param_group[
                        'lr_out'] = param_group['lr_out'] * self.gamma_out
                    logging.debug('Updated lr_out=', param_group['lr_out'])
