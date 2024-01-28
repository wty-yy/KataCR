from katacr.constants.label_list import ground_unit_list, flying_unit_list, tower_unit_list, other_unit_list, spell_unit_list, background_item_list
level2units = {
  0: ['blood', 'butterfly', 'flower', 'skull', 'cup'],
  1: ground_unit_list + tower_unit_list,
  2: flying_unit_list,
  3: other_unit_list + ['ribbon'],
}
unit2level = {unit: level for level, units in level2units.items() for unit in units}
unit2level['small-text'] = unit2level['big-text'] = unit2level['text']
drop_units = [
  'emote', 'small-text', 'elixir', 'bar', 'tower-bar',
  'king-tower-bar', 'clock', 'big-text', 'background-items',
  'bar-level'
]
drop_fliplr = ['text', 'bar', 'bar-level', 'king-tower-bar', 'tower-bar', 'elixir']
drop_box = background_item_list + ['bar-level']

background_size = (568, 896)
xyxy_grids = (6, 64, 562, 864)
bottom_center_grid_position = {
  'king1': (9, 4.7),
  'queen1_0': (3.5, 7.7),
  'queen1_1': (14.5, 7.7),
  'king0': (9, 30.5),
  'queen0_0': (3.5, 26.7),
  'queen0_1': (14.5, 26.7),
}

except_king_tower_unit_list = tower_unit_list.copy()
except_king_tower_unit_list.remove('king-tower')
component_prob = {x: 0.95 for x in except_king_tower_unit_list}
component_prob.update({'king-tower': 0.5})  # king-tower-bar
component_prob.update(  # the probability of adding a component
  {x: 0.2 for x in (ground_unit_list + flying_unit_list)}
)
important_components = [(('bar', 'bar-level'), 0.9)]  # highter prob to use important components, when add components.
component_cfg = {  # center [cell pos, top_center, bottom_center], dx_range, dy_range, width
  'small-text': ['top_center', (0, 0), (-1, -0.5), None],
  'elixir': ['bottom_center', (0, 0), (-2, 0), None],
  'bar': ['top_center', (0, 0), (-0.5, 0), None],
  'bar-level': ['top_center', (0, 0), (-0.5, 0), None],
  'bar': ['top_center', (0, 0), (0, 1), None],
  'bar-level': ['top_center', (0, 0), (0, 1), None],
  'tower-bar0': ['bottom_center', (0, 0), (-2, -1), (2.5, 3)],
  'tower-bar1': ['top_center', (0, 0), (0, 0.5), (2.5, 3)],
  'king-tower-bar0': ['bottom_center', (0, 0), (1, 1.5), (4.5, 5.5)],
  'king-tower-bar1': ['top_center', (0, 0), (0, 0), (4.5, 5.5)],
  'clock': ['bottom_center', (0, 0), (2, 1.5), None]
}
# (prob, [center, dx_range, dy_range, width_range, max_num]*n)
item_cfg = {
  'big-text': (0.05, [[(9, 13), (0, 0), (0, 5), None, 1]]),
  'emote': (0.1, [
    [(1, 3), (0, 0), (0, 28), (1.5, 2), 4],  # left range
    [(17, 3), (0, 0), (0, 28), (1.5, 2), 4],  # right range
    [(13, 32), (0, 1), (0, 0), (2.5, 3.5), 1],  # bottom range
    [(4.5, 2.5), (0, 1), (0, 0), (2.5, 3.5), 1],  # top range
  ]),
  'blood': (0.3, [[(9, 16), (-9, 9), (-8, 8), None, 30]]),  # center range
  'butterfly': (0.1, [[(0, 0), (0, 18), (0, 32), None, 3]]),  # all
  'flower': (0.3, [[(0, 0), (0, 18), (0, 32), None, 5]]),  # all
  'ribbon': (0.5, [[(0, 0), (0, 18), (0, 32), None, 50]]),  # all
  'skull': (0.05, [[(0, 0), (0, 18), (0, 32), None, 3]]),  # all
  'cup': (0.05, [[(0, 0), (0, 18), (0, 32), None, 4]]),  # all
}
except_spell_flying_unit_list = list(set(flying_unit_list) - set(spell_unit_list))
component2unit = {
  'small-text': ground_unit_list + flying_unit_list,
  'elixir': ground_unit_list + flying_unit_list,
  ('bar', 'bar-level'): ground_unit_list + except_spell_flying_unit_list,
  'tower-bar': except_king_tower_unit_list,
  'king-tower-bar': ['king-tower'],
  'clock': ground_unit_list + except_spell_flying_unit_list,
}

# Augmentation (mask and transparency)
background_augment = {
  'xyxy': (0,56,568,490),
  'prob': 0.5
}
aug2prob = {  # accumulate probablity
  'red': 0.1,
  'blue': 0.1,
  'golden': 0.2,
  'white': 0.1,
  'trans': 0.2,
}
aug2unit = {
  'red': ground_unit_list + tower_unit_list + except_spell_flying_unit_list,
  'blue': ground_unit_list + tower_unit_list + except_spell_flying_unit_list,
  'golden': ['text'] + ground_unit_list + except_spell_flying_unit_list,
  'white': ['clock'] + ground_unit_list + except_spell_flying_unit_list + tower_unit_list,
  'trans': ground_unit_list + except_spell_flying_unit_list
}
alpha_transparency = 150
color2RGB = {
  'red': (255, 0, 0),
  'blue': (0, 0, 255),
  'golden': (255, 215, 0),
  'white': (255, 255, 255)
}
color2alpha = {
  'red': 80,
  'blue': 100,
  'golden': 150,
  'white': 150
}
color2bright = {  # brightness range
  'red': (30, 50),
  'blue': (30, 80),
  'golden': (70, 80),
  'white': (110, 120),
}

unit_scale = {x: ((0.5, 1.2), 1.0) for x in ('elixir', 'clock')}
unit_stretch = {x: ((0.5, 0.8), 0.1) for x in (ground_unit_list + except_spell_flying_unit_list)}
tower_intersect_ratio_thre = 0.8
bar_intersect_ratio_thre = 0.1

grid_size = (18, 32)  # (x, y)
map_ground = [
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
  [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],  # king tower
  [0.5, 0.5, 0.5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0.5, 0.5, 0.5],
  [0.5, 0.5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.5],
  [0.5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0.5],
  [1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1],  # queen tower
  [1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1],
  [1, 1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1],
  [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],  # river
  [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],  # river
  [1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1],
  [1, 1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1],  # queen tower
  [1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1],
  [1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1],
  [0.5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0.5],  # king tower
  [0.5, 0.5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.5],
  [0.5, 0.5, 0.5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0.5, 0.5, 0.5],
  [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
]

map_fly = [
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
]