from katacr.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss': MeanMetric(),
    'loss_noobj': MeanMetric(),
    'loss_coord': MeanMetric(),
    'loss_obj': MeanMetric(),
    'loss_class': MeanMetric(),
    
    'epoch': 0,
    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'learning_rate': 0,
  },
  folder2name={
    'metrics': [
      'loss',
      'loss_noobj',
      'loss_coord',
      'loss_obj',
      'loss_class',
    ],
    'charts': [
      'SPS', 'SPS_avg',
      'epoch', 'learning_rate'
    ]
  }
)