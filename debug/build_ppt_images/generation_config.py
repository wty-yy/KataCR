from katacr.constants.label_list import ground_unit_list, flying_unit_list, tower_unit_list, other_unit_list, spell_unit_list, background_item_list, object_unit_list
ground_spell_list = list(set(ground_unit_list) & set(spell_unit_list))
ground_unit_except_spell_list = list(set(ground_unit_list) - set(spell_unit_list))
level2units = {
  0: ground_spell_list + ['blood', 'butterfly', 'flower', 'skull', 'cup', 'snow', 'grave', 'ruin', 'circle'],
  1: ground_unit_except_spell_list + tower_unit_list,
  2: flying_unit_list,
  3: other_unit_list + ['ribbon', 'scoreboard', 'crown-icon', 'king-tower-level'],
}
unit2level = {unit: level for level, units in level2units.items() for unit in units}
unit2level['small-text'] = unit2level['big-text'] = unit2level['text']
drop_units = [
  'emote', 'small-text', 'elixir', 'bar', 'tower-bar',
  'king-tower-bar', 'clock', 'big-text', 'background-items',
  'bar-level', 'skeleton-king-skill', 'skeleton-king-bar', # 'tesla-evolution-shock'
  'dagger-duchess-tower-bar',
]
drop_fliplr = [
  'text', 'bar', 'bar-level', 'king-tower-bar', 'tower-bar', 'elixir',
  'skeleton-king-bar', 'dagger-duchess-tower-bar', 'dagger-duchess-tower-icon',
  'king-tower-level', 'scoreboard'
]
drop_box = background_item_list + ['text']

background_size = (568, 896)
xyxy_grids = (6, 64, 562, 864)  # xyxy pixel size of the whold grid
towers_bottom_center_grid_position = {
  'king1': (9, 4.7),
  'queen1_0': (3.5, 7.7),
  'queen1_1': (14.5, 7.7),
  'king0': (9, 30.5),
  'queen0_0': (3.5, 26.7),
  'queen0_1': (14.5, 26.7),
}

except_king_tower_unit_list = tower_unit_list.copy()
except_king_tower_unit_list.remove('king-tower')
# Start component generation probability
component_prob = {x: 1.0 for x in (except_king_tower_unit_list + ['ruin', 'king-tower-ruin'])}  # defence tower and ruin
component_prob.update({'king-tower': 0.5})  # king-tower-bar
component_prob.update(  # the probability of adding a component
  {x: 0.4 for x in (ground_unit_list + flying_unit_list)}
)
important_components = [  # highter prob to use important components, when add components.
  (('bar', 'bar-level'), None),  # Modify in Generator._add_component(unit)
  # ('tesla-evolution-shock', 1.0),
  ('skeleton-king-skill', 1.0),
  ('skeleton-king-bar', 1.0),
]
option_components = [  # choose one option by probs
  (('king-tower-bar', 'king-tower-level'), (0.5, 0.5))
]
# center [cell pos | top_center | bottom_center], dx_range, dy_range, width, component generation format [bottom_center | center]
component_cfg = {
  'small-text': ['top_center', (0, 0), (-1, -0.5), None, 'bottom_center'],
  'elixir': ['bottom_center', (0, 0), (-2, -1), None, 'bottom_center'],
  'bar': ['top_center', (0, 0), (-0.2, 0.2), None, 'left_center'],  # add one bar-level at leftside
  'bar-level': ['top_center', (0, 0), (-0.2, 0.5), None, 'bottom_center'],
  'tower-bar0': ['bottom_center', (0, 0), (-1.5, -0.5), (2.5, 3), 'bottom_center'],
  'tower-bar1': ['top_center', (0, 0), (-1.0, -1.2), (2.5, 3), 'top_center'],
  'dagger-duchess-tower-bar0': ['bottom_center', (0.1, 0.2), (-0.5, 0), (2.5, 3), 'top_center'],
  'dagger-duchess-tower-bar1': ['top_center', (0.2, 0.2), (-1.2, -1.2), (2.5, 3), 'bottom_center'],
  'king-tower-bar0': ['bottom_center', (0, 0), (1, 1.5), (4.5, 5.5), 'bottom_center'],
  'king-tower-bar1': ['top_center', (0, 0), (0, 0), (4.5, 5.5), 'bottom_center'],
  'king-tower-level0': ['bottom_center', (0, 0), (1, 1.5), (4.5, 5.5), 'bottom_center'],
  'king-tower-level1': ['top_center', (0, 0), (0, 0), (4.5, 5.5), 'bottom_center'],
  'crown-icon': ['top_center', (-0.5, 0.5), (0.2, -0.5), (2, 3), 'bottom_center'],
  'clock': ['bottom_center', (0, 0), (2, 1.5), None, 'bottom_center'],
  # 'tesla-evolution-shock': ['center', (0, 0), (0, 0), None, 'center'],
  'skeleton-king-skill': ['center', (0, 0), (0, 0), None, 'center'],
  'skeleton-king-bar': ['top_center', (0, 0), (-0.5, -0.3), None, 'bottom_center'],
}
except_spell_unit_list = list(set(ground_unit_list).union(flying_unit_list) - set(spell_unit_list))
except_object_unit_list = list(set(ground_unit_list).union(flying_unit_list) - set(object_unit_list))
except_spell_and_object_unit_list = list(set(ground_unit_list).union(flying_unit_list) - set(spell_unit_list) - set(object_unit_list))
component2unit = {  # the component below to units, prob
  'small-text': (except_object_unit_list, 0.0),
  'elixir': (except_spell_and_object_unit_list, 1/2),
  ('bar', 'bar-level'): (except_spell_and_object_unit_list, 1.0),
  'tower-bar': (except_king_tower_unit_list, 1.0),
  'dagger-duchess-tower-bar': (['dagger-duchess-tower'], 1.0),
  ('king-tower-bar', 'king-tower-level'): (['king-tower'], 1.0),
  'crown-icon': (['ruin'], 0.1),
  'crown-icon': (['king-tower-ruin'], 0.1),
  'clock': (ground_unit_list + except_spell_and_object_unit_list + ['bomb'], 1/2),
  # 'tesla-evolution-shock': (['tesla-evolution'], 1.0),
  'skeleton-king-skill': (['skeleton-king'], 1.0),
  'skeleton-king-bar': (['skeleton-king'], 1.0),
}
bar_xy_range = (-0.3, -0.1)  # (width(bar-level) - width(bar)) / 2

# (prob, [bottom_center, dx_range, dy_range, width_range, max_num]*n)
item_cfg = {
  'big-text': (1.01, [[(9, 13), (0, 0), (0, 5), None, 1]]),  # center
  'small-text': (0.02, [[(0, 0), (0, 18), (0, 32), None, 2]]),  # all
  'emote': (1.1, [
    [(1, 3), (0, 0), (0, 28), (1.5, 2), 4],  # left range
    [(17, 3), (0, 0), (0, 28), (1.5, 2), 4],  # right range
    [(13, 32), (0, 1), (0, 0), (2.5, 3.5), 1],  # bottom range
    [(4.5, 2.5), (0, 1), (0, 0), (2.5, 3.5), 1],  # top range
  ]),
  'blood': (1.3, [[(9, 16), (-9, 9), (-8, 8), None, 30]]),  # center range
  'butterfly': (1.1, [[(0, 0), (0, 18), (0, 32), None, 3]]),  # all
  'flower': (0.3, [[(0, 0), (0, 18), (0, 32), None, 5]]),  # all
  'ribbon': (0.5, [[(0, 0), (0, 18), (0, 32), None, 30]]),  # all
  'skull': (0.05, [[(0, 0), (0, 18), (0, 32), None, 3]]),  # all
  'cup': (0.05, [[(0, 0), (0, 18), (0, 32), None, 4]]),  # all
  'snow': (1.05, [[(0, 0), (0, 18), (0, 32), None, 4]]),  # all
  'grave': (1.05, [[(0, 0), (0, 18), (0, 32), None, 20]]),  # all
  'scoreboard0': (1.01, [[(17.5, 21.2), (0, 0), (0, 0), None, 1]]),  # right down
  'scoreboard1': (1.01, [[(17.5, 14.2), (0, 0), (0, 0), None, 1]]),  # right up
  'crown-icon': (0.01, [[(0, 0), (0, 18), (0, 32), (2, 3), 4]]),  # all
  'circle': (0.05, [[(0, 0), (0, 18), (0, 32), None, 4]]),  # all
}

# Augmentation (mask and transparency)
background_augment = {
  'xyxy': (0,56,568,490),
  'prob': 0.5
}
aug2prob = {  # accumulate probablity
  'red': 0.02,    # 0.05
  'blue': 0.02,   # 0.05
  'golden': 0.02, # 0.05
  'white': 0.02,  # 0.05
  'violet': 0.01,  # 0.02
  'trans': 0.00,  # 0.05
}

aug2unit = {
  'red': tower_unit_list + except_spell_and_object_unit_list,
  'blue': tower_unit_list + except_spell_and_object_unit_list,
  'golden': ['text'] + except_spell_and_object_unit_list,
  'white': ['clock'] + except_spell_and_object_unit_list + tower_unit_list,
  'violet': except_spell_and_object_unit_list + tower_unit_list,
  'trans': except_spell_and_object_unit_list
}
alpha_transparency = 150
color2RGB = {
  'red': (255, 0, 0),
  'blue': (0, 0, 255),
  'golden': (255, 215, 0),
  'white': (255, 255, 255),
  'violet': (127, 0, 255),
}
color2alpha = {
  'red': 80,
  'blue': 100,
  'golden': 150,
  'white': 150,
  'violet': 100,
}
color2bright = {  # brightness range
  'red': (30, 50),
  'blue': (30, 80),
  'golden': (70, 80),  # (70, 80)
  'white': (110, 120),  # (110, 120)
  'violet': (10, 30),
}

# unit_scale = {x: ((0.5, 1.2), 1.0) for x in ('elixir', 'clock')}
unit_scale = {x: ((0.5, 1.0), 1.0) for x in ('elixir', 'clock')}
unit_stretch = {x: ((0.5, 0.8), 0.0) for x in (except_spell_and_object_unit_list)}
tower_intersect_ratio_thre = 0.8
bar_intersect_ratio_thre = 0.5
king_tower_generation_ratio = 0.95  # generate king-tower ruin 1-prob
tower_generation_ratio = {  # generate tower ruin 1 - sum(probs)
  'queen-tower': 0.25,
  'cannoneer-tower': 0.25,
  'dagger-duchess-tower': 0.25,
}

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