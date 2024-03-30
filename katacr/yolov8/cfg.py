from katacr.constants.label_list import unit2idx
unit_nums = 30
map_update_mode = 'dynamic'
intersect_ratio_thre = 0.6
train_datasize = 20000
img_size = (576, 896)  # width, height
assert img_size[0] % 32 == 0 and img_size[1] % 32 == 0, "The shape must be 32 multiple"

base_idxs = 13
max_detect_num = 65
detector1_list = list(range(base_idxs)) + list(range(base_idxs, max_detect_num))
detector2_list = list(range(base_idxs)) + list(range(max_detect_num, len(unit2idx)))
detection_range = {  # detector_name: idx_list
  'detector1': detector1_list,
  'detector2': detector2_list,
}
invalid_units = ['selected', 'mirror']
for n in invalid_units:
  idx = unit2idx[n]
  for k in detection_range.keys():
    if idx in detection_range[k]:
      detection_range[k].remove(idx)

if __name__ == '__main__':
  print(detection_range)
  total = 0
  for k, v in detection_range.items():
    print(f"{k} detect classs number: {len(v)}")
    total += len(v)
  print("Summery:", total)