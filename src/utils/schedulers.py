#src/utils/schedulers.py
import math
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup and cosine annealing scheduler.

    Has to be used in combination with `src.callbacks.step_lr.StepLRCallback`.

    Modifies the learning rate every gradient step, but is configured based on epochs.
    """

    def __init__(
        self, optimizer, warmup_epochs, max_epochs, steps_per_epoch=1, min_lr=0.0, last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.set_steps_per_epoch(steps_per_epoch)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.max_steps = self.max_epochs * self.steps_per_epoch

    def get_lr(self):
        # The scheduler treats every gradient update as a step if we set
        # lr_scheduler.interval="step" in LighttningModule.configure_optimizers()
        current_step = self.last_epoch + 1

        if current_step <= self.warmup_steps:
            # Linear warmup
            return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class ExpDecayLR(_LRScheduler):
    """Based on optax.schedules.exponential_decay.
    https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#exponential-decay-schedule
    """
    def __init__(self, optimizer, init_value, transition_steps, decay_rate, transition_begin=0, end_value=0.0):
        """
        rate_factor = ((count - transition_begin) / transition_steps)
        decayed_value = init_value * (decay_rate ** rate_factor)

        Args:
        init_value:     the initial learning rate.
        transition_steps:   must be positive. See the decay computation above.
        decay_rate:     must not be zero. The decay rate.
        transition_begin:   must be positive. After how many steps to start annealing (before this many steps the scalar value is held fixed at init_value).
        end_value:  the value at which the exponential decay stops. When decay_rate < 1, end_value is treated as a lower bound, otherwise as an upper bound. Has no effect when decay_rate = 0.
        steps_per_epoch:    number of steps per epoch - depends dataset size and batch size. If not provided, it will be set to 1.
        """
        self.init_value = init_value
        self.transition_steps = transition_steps
        self.decay_rate = decay_rate
        self.transition_begin = transition_begin
        self.end_value = end_value
        super().__init__(optimizer)

    def _body_fn(self, count):
        rate_factor = ((count - self.transition_begin) / self.transition_steps)
        decayed_value = self.init_value * (self.decay_rate ** rate_factor)
        return max(decayed_value, self.end_value)
    
    def get_lr(self):
        current_step = self.last_epoch
        return [self._body_fn(current_step)]