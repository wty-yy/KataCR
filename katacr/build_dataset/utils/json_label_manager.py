import json, numpy as np
from pathlib import Path
path = Path("/home/wty/Coding/datasets/CR/images/part2/OYASSU_20230212_episodes/4/00000.json")

convert_dict = {
  # 'lumberjack': 'lumberjack1'
}
xy2idx = {'x0': 0, 'y0': 1, 'x1': 2, 'y1': 3}
remove_list = [
  # (None, {'x0': (390, 600), 'y1': (0, 41)}),
  # ('king-tower-bar1', {'x0': (210,220)})
]
add_list = [
  # ('king-tower-bar1', (212.5874125874126, 1.3986013986013988, 354.54545454545456, 38.46153846153846))
]
json_range = [  # process json file range, could count the unit number
  # (0, 1650),
  # (1860, 2175),
  # (2775, 3135),
  # (3975, 4275)
]
update_count = {}
remove_count = {}
add_count = {}
unit_count = {}

def solve(path: Path):
  with open(path, 'r') as file:
    data = json.load(file)
  include = [0] * len(add_list)
  for i, box in enumerate(data['shapes']):
    for k1, k2 in convert_dict.items():
      if k1 in box['label']:
        print(f"Move {k1} in {path} to {k2}")
        box['label'] = box['label'].replace(k1, k2)
        if k1 not in update_count: update_count[k1] = 0
        update_count[k1] += 1
    for name, cfg in remove_list:
      if name is None or name == box['label']:
        flag = True
        xyxy = np.array(box['points']).reshape(-1)
        for xy, r in cfg.items():
          tmp = xyxy[xy2idx[xy]]
          if not r[0] <= tmp <= r[1]:
            flag = False
        if flag:
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
  for box in data['shapes']:
    label = box['label']
    if label not in unit_count: unit_count[label] = 0
    unit_count[label] += 1
  with open(path, 'w') as file:
    json.dump(data, file, indent=2)
    
if __name__ == '__main__':
  # path_dir = Path("/home/wty/Coding/datasets/CR/images/part2/OYASSU_20230212_episodes/4")
  path_dir = Path("/home/wty/Coding/datasets/CR/images/part2/OYASSU_20210528_episodes/6")
  process_count = 0
  for path in sorted(list(path_dir.glob("*.json"))):
    name = path.stem
    flag = len(json_range) == 0
    for l, r in json_range:
      if l <= int(name) <= r:
        flag = True
    if flag:
      process_count += 1
      solve(path)
  print("Update count:", update_count)
  print("Remove count:", remove_count)
  print("Add count:", add_count)
  unit_count = dict(sorted(unit_count.items(), key=lambda x: (x[0][-1], x[0])))
  print(f"Unit count (img num={process_count}):", unit_count)

