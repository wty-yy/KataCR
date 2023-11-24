import time
import numpy as np
from tensorboardX import SummaryWriter

class MeanMetric():

    count: int = 0
    mean: float = 0

    def update_state(self, value):
        try:
            value = float(value)
        except:
            if len(value) == 0: return
            value = np.mean(value)
        self.count += 1
        self.mean += (value - self.mean) / self.count
    
    def reset_state(self):
        self.count, self.mean = 0, 0.
    
    def result(self):
        return self.mean


class Logs:
    """
    The logs during 'one episode' of the Agent.
    
    -   support_type (list): The type that logs support to update

    Initialize:
    -   init_logs (dict): What logs you want to memory, such like:
            init_logs = {
                'episode': 0,
                'step': 0,
                'q_value': keras.metrics.Mean(name='q_value'),
                'loss': keras.metrics.Mean(name='loss'),
                'frame': []
            }
    -   folder2name (dict): match name with the cls folder, such like:
            folder2name = {
                'charts': ['episode_step', 'episode_return'],
                'metrics': ['q_value', 'loss']
            }
    
    Function
    -   reset(): reset the logs, use it after one episode.

    -   update(keys:list, values:list):
            update 'zip(keys, values)' one by one in logs.

    -   to_dict(drops:list):
            Cast 'logs' to 'dict' for saving to 'utils/History' class.
            -   drops: The keys in logs that you don't want to return.
    """
    support_type = [
        int,
        float,
        list,
        MeanMetric
    ]

    def __init__(self, init_logs:dict, folder2name:dict):
        self.start_time, self.folder2name = time.time(), folder2name
        # type check
        for key, value in init_logs.items():
            flag = False
            for type in self.support_type:
                if isinstance(value, type):
                    flag = True; break
            if not flag:
                raise Exception(
                    f"Error: Don't know {key}'s type '{type(value)}'!"
                )
        count = 0
        self.logs = init_logs
        for _, names in folder2name.items():
            count += len(names)
            for name in names:
                if name not in self.logs.keys():
                    raise Exception(f"Error: The name '{name}' is not in 'self.logs' {self.logs}")
        if count != len(self.logs.keys()):
            raise Exception(f"Error: Some name is not in folder")
    
    def reset(self):
        self.start_time = time.time()
        # BUGFIX:
        #   Can't use self.logs = self.init_logs.copy()
        #   'keras.metrics.Metric' and 'list' will not be reset
        for key, value in self.logs.items():
            if isinstance(value, list):
                self.logs[key] = []
            elif isinstance(value, MeanMetric):
                value.reset_state()
            else:  # int or float
                self.logs[key] = 0
    
    def update(self, keys:list, values:list):
        for key, value in zip(keys, values):
            if value is not None:
                target = self.logs[key]
                if isinstance(target, MeanMetric):
                    target.update_state(value)
                elif isinstance(target, list):
                    target.append(value)
                else: # int or float or None
                    self.logs[key] = value

    def to_dict(self, drops:list=None):
        ret = self.logs.copy()
        for key, value in ret.items():
            if value is not None:
                target = self.logs[key]
                if isinstance(target, MeanMetric):
                    ret[key] = target.result() if target.count else None
        if drops is not None:
            for drop in drops:
                ret.pop(drop)
        return ret
    
    def get_time_length(self):
        return time.time() - self.start_time
    
    def writer_tensorboard(self, writer:SummaryWriter, global_step, drops=[]):
        d = self.to_dict()
        for folder, names in self.folder2name.items():
            for name in names:
                if name in drops: continue
                if d[name] is not None:
                    writer.add_scalar(folder+'/'+name, d[name], global_step)
    