import json, numpy as np
from pathlib import Path

xy2idx = {'x0': 0, 'y0': 1, 'x1': 2, 'y1': 3}
convert_dict = {  # target label: (origin label, position range)
  # 'ice-spirit-evolution-symbol1': ('ice-spirit-evoltuion-symbol1', {}),
  'dagger-duchess-tower1': ('queen-tower1', {}),
  'cannoneer-tower0': ('queen-tower0', {}),
}
remove_list = [  # (label[str | None], remove box range[dict]), None means any label satisfied the box range.
  # ('ice-spirit-evolution-symbol1', {})
  # (None, {'y0': (826, 10000)})
  # (None, {'x0': (390, 600), 'y1': (0, 64)}),
  # ('king-tower-bar1', {'x0': (210,220)}),
  # (None, {'x0': (480, 500), 'x1': (530, 550), 'y0': (50, 70), 'y1': (90, 110)}),  # wrong box around top-right elixir text
  # (None, {'x0': (390, 1000), 'y1': (0, 120)}),
]
add_list = [  # (label, xyxy)
  # ('king-tower-bar1', (212.5874125874126, 1.3986013986013988, 354.54545454545456, 38.46153846153846))
]
json_range = [  # process json file range, could count the unit number
]
jpg_range = [
  # WTY_20240222_8spells
  # (569,655),(2927,3010),( 571,600),(1984,2015),(3127,3180),( 613,637),(1837,1862),( 785,901),(2388,2505),(3671,3787),( 943,1126),( 1053,1070),(2137,2155),( 1327,1365),(2676,2715),(3892,3940),( 1590,1650),(3345,3398)
  # (0,1905),(2400,3450),(3600,4230)
  # (0, 2655), (2880, 3480)
  # (0, 3360), (3495, 4065)
]
REMOVE_EXTRA_FILES = False
debug_list = [  # print filepaths when belowing labels in
  # 'elixir0'
]
delta_list = [
  # ('queen-tower0', 2)  # delta
]
update_count = {}
remove_count = {}
add_count = {}
unit_count = {}

def check_position(xyxy, cfg):
  flag = True
  for xy, r in cfg.items():
    tmp = xyxy[xy2idx[xy]]
    if not r[0] <= tmp <= r[1]:
      flag = False
  return flag

def solve(path: Path):
  with open(path, 'r') as file:
    data = json.load(file)
  include = [0] * len(add_list)
  for i, box in enumerate(data['shapes']):
    xyxy = np.array(box['points']).reshape(-1)
    for l2, (l1, cfg) in convert_dict.items():
      if l1 == box['label'] and check_position(xyxy, cfg):
        print(f"Move {l1} in {path} to {l2}")
        box['label'] = box['label'].replace(l1, l2)
        if l1 not in update_count: update_count[l1] = 0
        update_count[l1] += 1
    for name, cfg in remove_list:
      if name is None or name == box['label']:
        if check_position(xyxy, cfg):
          # s = input(f"Remove label {bbox['label']} at {xyxy} ({path})? [Enter(yes)/no]")
          # if s in ['no', 'No']:
          #   continue
          data['shapes'].pop(i)
          if name not in remove_count: remove_count[name] = 0
          remove_count[name] += 1
    for i, (name, cfg) in enumerate(add_list):
      if name == box['label']:
        include[i] += 1
  for i, (name, cfg) in enumerate(add_list):
    if include[i]: continue
    data['shapes'].append(
      {
        "label": name,
        "points": [
          [
            cfg[0],
            cfg[1]
          ],
          [
            cfg[2],
            cfg[3]
          ]
        ],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {}
      }
    )
    if name not in add_count: add_count[name] = 0
    add_count[name] += 1
  origin_count = unit_count.copy()
  for box in data['shapes']:
    label = box['label']
    if label not in unit_count: unit_count[label] = 0
    unit_count[label] += 1
    if label in debug_list:
      print(f"{label} in path: {path}")
  for name, delta in delta_list:
    if unit_count.get(name, 0) - origin_count.get(name, 0) != delta:
      print(f"Wrong in delta count check! '{name}' in {path} don't have {delta} with previous one.")
  with open(path, 'w') as file:
    json.dump(data, file, indent=2)

def remove_file(path: Path):
  name = path.stem
  json_path = path.with_name(name+'.json')
  json_path.unlink(missing_ok=True)
  img_path = path.with_name(name+'.jpg')
  img_path.unlink(missing_ok=True)
  print(f"Remove {json_path}, {img_path}")
    
if __name__ == '__main__':
  path_dir = Path("/home/yy/Coding/datasets/Clash-Royale-Dataset/images/part2/WTY_20240412/dagger1_cannoneer0_evolution_ram")
  process_count = 0
  print("Resolve directory:", path_dir)
  if REMOVE_EXTRA_FILES:
    s = input("Are you sure to remove the extra files which not in 'json_range'? [Yes(Enter)|No] ")
    if s.lower() in ['no', 'n', 'false']: exit()
  for path in sorted(list(path_dir.glob("*.json"))):
    name = path.stem
    flag = len(json_range) == 0
    for l, r in json_range:
      if l <= int(name) <= r:
        flag = True
    if flag:
      process_count += 1
      solve(path)
    elif REMOVE_EXTRA_FILES:
      remove_file(path)
  for path in sorted(list(path_dir.glob("*.jpg"))):
    id = int(path.stem)
    flag = len(jpg_range) == 0
    for l, r in jpg_range:
      if l <= id <= r:
        flag = True
    if not flag and REMOVE_EXTRA_FILES:
      remove_file(path)
  print("Update count:", update_count)
  print("Remove count:", remove_count)
  print("Add count:", add_count)
  unit_count = dict(sorted(unit_count.items(), key=lambda x: (x[0][-1], x[0])))
  print(f"Unit count (img num={process_count}):", unit_count)

