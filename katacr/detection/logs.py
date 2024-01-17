from katacr.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'loss_obj_train': MeanMetric(),
    'loss_box_train': MeanMetric(),
    'loss_cls_train': MeanMetric(),

    'loss_val': MeanMetric(),
    'loss_obj_val': MeanMetric(),
    'loss_box_val': MeanMetric(),
    'loss_cls_val': MeanMetric(),

    'P@50_val': MeanMetric(),
    'R@50_val': MeanMetric(),
    'AP@50_val': MeanMetric(),
    'AP@75_val': MeanMetric(),
    'mAP_val': MeanMetric(),

    'iou': MeanMetric(),
    'ciou': MeanMetric(),
    '1-ciou': MeanMetric(),
    'num_pos': MeanMetric(),
    
    'epoch': 0,
    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'learning_rate': 0,
    'learning_rate_bias': 0,
  },
  folder2name={
    'metrics/train': [
      'loss_train',
      'loss_obj_train',
      'loss_box_train',
      'loss_cls_train',
    ],
    'metrics/val': [
      'P@50_val',
      'R@50_val',
      'AP@50_val',
      'AP@75_val',
      'mAP_val',
      'loss_val',
      'loss_obj_val',
      'loss_box_val',
      'loss_cls_val',
    ],
    'metrics/debug': [
      'iou',
      'ciou',
      '1-ciou',
      'num_pos',
    ],
    'charts': [
      'SPS', 'SPS_avg',
      'epoch', 'learning_rate', 'learning_rate_bias'
    ]
  }
)