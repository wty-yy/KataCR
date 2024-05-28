from katacr.constants.label_list import unit2idx
### Generation configuration ###
unit_nums = 40  # generate units number
map_update_mode = 'dynamic'  # generate sample strategy
intersect_ratio_thre = 0.5  # maximum coverage ratio

train_datasize = 20000
img_size = (576, 896)  # width, height
assert img_size[0] % 32 == 0 and img_size[1] % 32 == 0, "The shape must be 32 multiple"

base_idxs = 15  # All detector requires to detect indexs in `constants/label_list.py`, indexs range in `1...base_idxs`
### Three Detector ###
# max_detect_num = 65  # 160  # 85  # 65
# num_detector = 3  # 1  # 2  # 3
# noise_unit_ratio = 1/4
### Two Detector ###
max_detect_num = 85  # 160  # 85  # 65
num_detector = 2  # 1  # 2  # 3
noise_unit_ratio = 1/4
### One Detector ###
# max_detect_num = 160  # 160  # 85  # 65
# num_detector = 1  # 1  # 2  # 3
# noise_unit_ratio = 0
detector1_list = list(range(base_idxs)) + list(range(base_idxs, max_detect_num))
detector2_list = list(range(base_idxs)) + list(range(max_detect_num, len(unit2idx)))
detection_range = {  # detector_name: idx_list
  'detector1': detector1_list,
  'detector2': detector2_list,
}
invalid_units = ['selected', 'text', 'mirror', 'tesla-evolution-shock', 'zap-evolution']
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