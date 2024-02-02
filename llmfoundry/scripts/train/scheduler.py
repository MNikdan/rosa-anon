from typing import TYPE_CHECKING, List, Union
from composer.optim.scheduler import ComposerScheduler, _convert_time
from composer.core import State, Time
import warnings, textwrap
import numpy as np

# class DelayedLinearScheduler(ComposerScheduler):
#     def __init__(self, t_start: Union[str, Time], alpha_i: float = 1.0, alpha_f: float = 0.0, t_max: Union[str, Time] = '1dur'):
#         self.t_start = t_start
#         self.alpha_i = alpha_i
#         self.alpha_f = alpha_f
#         self.t_max = Time.from_timestring(t_max) if isinstance(t_max, str) else t_max

#     def __call__(self, state: State, ssr: float = 1.0):
#         t_start = _convert_time(self.t_start, state)
#         t_max = _convert_time(self.t_max, state, ssr=ssr)
#         current_time = state.timestamp.get(t_max.unit)

#         if current_time < t_start:
#             return 0.0
        
#         frac_of_total = min(1.0, ((current_time - t_start) / (t_max - t_start)).value)
#         current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)
#         return current_factor

class DelayedLinearWithWarmupScheduler(ComposerScheduler):
    def __init__(self,
                 t_start: Union[str, Time],
                 t_warmup: Union[str, Time],
                 alpha_i: float = 1.0,
                 alpha_f: float = 0.0,
                 t_max: Union[str, Time] = '1dur',
                 scale_warmup: bool = False):
        self.t_start = t_start
        self.t_warmup = t_warmup
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        # self.warmup_scheduler = DelayedLinearScheduler(t_start=t_start, alpha_i=0.0, alpha_f=alpha_i, t_max=t_start+t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        # _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)
        t_start = _convert_time(self.t_start, state)
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))
        
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_warmup.unit)

        if current_time < t_start:
            return 0.0
        elif current_time < t_start + t_warmup:
            frac_of_total = ((current_time - t_start) / t_warmup).value
            frac_of_total = min(1.0, frac_of_total)
            current_factor = frac_of_total * self.alpha_i
            return current_factor
            # if self.scale_warmup:
            #     return self.warmup_scheduler(state, ssr)
            # return self.warmup_scheduler(state)
        else:
            frac_of_total = ((current_time - t_start - t_warmup) / (t_max - t_start - t_warmup)).value if (t_max > t_start + t_warmup) else 0.0
            frac_of_total = min(1.0, frac_of_total)
            current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)
            return current_factor

class SchedulerSeriesTimed(ComposerScheduler):
    def __init__(self, schedulers, t_starts):
        self.schedulers = schedulers
        self.t_starts = t_starts

    def __call__(self, state: State, ssr: float = 1.0):
        t_starts = [_convert_time(t_start, state) for t_start in self.t_starts]
        current_time = state.timestamp
        
        current_scheduler = None
        max_t_start = -1
        for scheduler, t_start in zip(self.schedulers, t_starts):
            if t_start > max_t_start and current_time >= t_start:
                current_scheduler = scheduler
                max_t_start = t_start
        
        if current_scheduler is None:
            return 0.0
    
        return current_scheduler(state, ssr=ssr)

def get_lora_warmup_schedule(scheduler_config, spa_start):
    # create a SchedulerSeriesTimed
    alpha_i = scheduler_config.get('alpha_i', 1.0)
    alpha_f = scheduler_config.get('alpha_f', 0.0)
    t_warmup = scheduler_config.get('t_warmup', 0)
    alpha_i2 = scheduler_config.get('alpha_i2', alpha_i)
    schedulers = [
        DelayedLinearWithWarmupScheduler(t_start='0ba', t_warmup=t_warmup, alpha_i=alpha_i, alpha_f=alpha_f, t_max='1dur'),
        DelayedLinearWithWarmupScheduler(t_start=f'{spa_start}ba', t_warmup=t_warmup, alpha_i=alpha_i2, alpha_f=alpha_f, t_max='1dur')
    ]
    t_starts = ['0ba', f'{spa_start}ba']
    return SchedulerSeriesTimed(schedulers, t_starts)