"""
2023/11/01: Total unit numbers: 125 + 9 = 134
2023/11/11: Add little-prince, royal-guradian and archer-evolution, total: 137
"""
unit_list = [
  # object
  'king-tower',
  'queen-tower',
  # 2024/1/1: add
  'cannoneer-tower',
  # 2024/4/1: add
  'dagger-duchess-tower',
  # 2024/4/12: add
  'dagger-duchess-tower-bar',
  'tower-bar',
  'king-tower-bar',
  'bar',
  # 2024/3/10: add
  'bar-level',
  'clock',
  'emote',
  'text',
  'elixir',
  'selected',
  # 2024/3/7: add
  'skeleton-king-bar',
  # ==========
  'skeleton',
  'skeleton-evolution',
  'electro-spirit',  # add 'freeze' state
  'fire-spirit',
  'ice-spirit',  # add 'freeze' state
  'heal-spirit',  # add 'heal' state
  'goblin',
  'spear-goblin',
  'bomber',
  'bat',
  'bat-evolution',
  'zap',  # 'attack', add 'freeze' state
  'giant-snowball',  # 'norm/attack'
  'ice-golem',
  'barbarian-barrel',
  'barbarian',
  'barbarian-evolution',  # 'norm/rage'
  'wall-breaker',
  'rage',  # 'attack', add 'rage' state
  'the-log',  # 'attack'
  'archer',
  'arrows',  # 'norm/attack'
  'knight',
  'knight-evolution',
  'minion',
  'cannon',
  'skeleton-barrel',
  'firecracker',
  'firecracker-evolution',
  'royal-delivery',
  'royal-recruit',  # 'bare/shield'
  'royal-recruit-evolution',  # 'bare/shield', new 'dash'
  'tombstone',
  'mega-minion',
  'dart-goblin',
  'earthquake',  # 'norm/attack'
  'elixir-golem-big',
  'elixir-golem-mid',
  'elixir-golem-small',
  'goblin-barrel',
  'guard',  # new 'shield'
  'clone',  # 'attack', add 'clone' state
  'tornado',  # 'attack'
  'miner',
  'dirt',
  'princess',
  'ice-wizard',
  'royal-ghost',  # 'visible/invisible'
  'bandit',  # new 'dash'
  'fisherman',
  'skeleton-dragon',
  'mortar',
  'mortar-evolution',
  'tesla',  # 'norm/attack' as 'under/up'
  'fireball',  # 'norm/attack'
  'mini-pekka',
  'musketeer',
  'goblin-cage',
  'goblin-brawler',
  'valkyrie',
  'battle-ram',  # new 'dash'
  # 2024/4/1: add
  'battle-ram-evolution',
  'bomb-tower',
  'bomb',
  'flying-machine',
  'hog-rider',
  'battle-healer',  # add 'heal' state
  'furnace',
  'zappy',
  'baby-dragon',
  'dark-prince',  # 'bare/shield', new 'dash'
  'freeze',  # 'attack', add 'freeze' state
  'poison',  # 'attack', add 'poison' state
  'hunter',
  'goblin-drill',
  'electro-wizard',  # addd 'freeze' state
  'inferno-dragon',
  'phoenix-big',
  'phoenix-egg',
  'phoenix-small',
  'magic-archer',
  'lumberjack',
  'night-witch',
  'mother-witch',
  'hog',
  'golden-knight',  # new 'dash'
  'skeleton-king',
  'mighty-miner',
  'rascal-boy',
  'rascal-girl',
  'giant',
  'goblin-hut',
  'inferno-tower',
  'wizard',
  'royal-hog',
  'witch',
  'balloon',
  'prince',  # new 'dash'
  'electro-dragon',
  'bowler',
  # 'bowl',
  'executioner',
  'axe',
  'cannon-cart',  # new 'shield'
  'ram-rider',  # new 'dash'
  'graveyard',
  'archer-queen',  # 'visible/invisible'
  'monk',  # new 'shield'
  'royal-giant',
  'royal-giant-evolution',
  'elite-barbarian',
  'rocket',  # 'norm/attack'
  'barbarian-hut',
  'elixir-collector',
  'giant-skeleton',
  'lightning',  # 'norm/attack', add 'freeze' state
  'goblin-giant',
  'x-bow',
  'sparky',  # 'charge/over'
  'pekka',
  'electro-giant',
  'mega-knight',  # new 'dash'
  'lava-hound',
  'lava-pup',
  'golem',
  'golemite',
  # 2023/11/11: add
  'little-prince',
  'royal-guardian',
  'archer-evolution',
  # 2023/11/29: add
  'ice-spirit-evolution',
  # 2024/1/1: add
  'valkyrie-evolution',
  # 2024/2/1: add
  'bomber-evolution',
  'wall-breaker-evolution',
  # 2024/2/19: add
  'evolution-symbol',
  # 2024/2/22: forget
  'mirror',
  # 2024/3/4: add
  'tesla-evolution',
  # 2024/3/6: add
  'goblin-ball',
  # 2024/3/7: add
  'skeleton-king-skill', 'tesla-evolution-shock',
  # 2024/3/9: add
  'ice-spirit-evolution-symbol',
  # 2024/3/28: add
  'zap-evolution',
]

idx2unit = dict(enumerate(unit_list))
unit2idx = {name: idx for idx, name in enumerate(unit_list)}

ground_unit_list = [
  'archer',
  'archer-evolution',
  'archer-queen',
  'balloon',
  'bandit',
  'barbarian',
  'barbarian-evolution',
  'barbarian-barrel',
  'barbarian-hut',
  'battle-healer',
  'battle-ram',
  'battle-ram-evolution',
  'bomb',
  'bomb-tower',
  'bomber',
  'bomber-evolution',
  'bowler',
  'cannon',
  'cannon-cart',
  'clone',
  'dark-prince',
  'dart-goblin',
  'dirt',
  'earthquake',
  'electro-dragon',
  'electro-giant',
  'electro-spirit',
  'electro-wizard',
  'elite-barbarian',
  'elixir-collector',
  'elixir-golem-big',
  'elixir-golem-mid',
  'elixir-golem-small',
  'executioner',
  'fire-spirit',
  'firecracker',
  'firecracker-evolution',
  'fisherman',
  'freeze',
  'furnace',
  'giant',
  'giant-skeleton',
  'goblin',
  'goblin-brawler',
  'goblin-cage',
  'goblin-drill',
  'goblin-giant',
  'goblin-hut',
  'golden-knight',
  'golem',
  'golemite',
  'graveyard',
  'guard',
  'heal-spirit',
  'hog',
  'hog-rider',
  'hunter',
  'ice-golem',
  'ice-spirit',
  'ice-spirit-evolution',
  'ice-wizard',
  'inferno-tower',
  'knight',
  'knight-evolution',
  'lava-hound',
  'lava-pup',
  'little-prince',
  'lumberjack',
  'magic-archer',
  'mega-knight',
  'mighty-miner',
  'miner',
  'mini-pekka',
  'monk',
  'mortar',
  'mortar-evolution',
  'mother-witch',
  'musketeer',
  'night-witch',
  'pekka',
  'phoenix-egg',
  'poison',
  'prince',
  'princess',
  'rage',
  'ram-rider',
  'rascal-boy',
  'rascal-girl',
  'royal-delivery',
  'royal-ghost',
  'royal-giant',
  'royal-giant-evolution',
  'royal-guardian',
  'royal-hog',
  'royal-recruit',
  'royal-recruit-evolution',
  'skeleton',
  'skeleton-evolution',
  'skeleton-king',
  'skeleton-king-skill',
  'sparky',
  'spear-goblin',
  'tesla',
  'tesla-evolution',
  'tesla-evolution-shock',
  'tombstone',
  'valkyrie',
  'valkyrie-evolution',
  'wall-breaker',
  'wall-breaker-evolution',
  'witch',
  'wizard',
  'the-log',
  'x-bow',
  'zappy',
]
tower_unit_list = [
  'king-tower',
  'queen-tower',
  'cannoneer-tower',
  'dagger-duchess-tower',
]
flying_unit_list = [
  'arrows',
  'axe',
  'baby-dragon',
  'bat',
  'bat-evolution',
  'flying-machine',
  'fireball',
  'giant-snowball',
  'goblin-ball',
  'goblin-barrel',
  'inferno-dragon',
  'lightning',
  'mega-minion',
  'minion',
  'phoenix-big',
  'phoenix-small',
  'rocket',
  'skeleton-barrel',
  'skeleton-dragon',
  'tornado',
  'zap',
]
spell_unit_list = [
  'arrows',
  'clone',
  'earthquake',
  'fireball',
  'freeze',
  'giant-snowball',
  'goblin-barrel',
  'graveyard',
  'lightning',
  'poison',
  'rage',
  'rocket',
  'skeleton-king-skill',
  'tesla-evolution-shock',
  'tornado',
  'zap',
]
other_unit_list = [
  'bar',
  'bar-level',
  'clock',
  'dagger-duchess-tower-bar',
  'elixir',
  'emote',
  'evolution-symbol',
  'ice-spirit-evolution-symbol',
  'king-tower-bar',
  'skeleton-king-bar',
  'text',
  'tower-bar',
]
background_item_list = [
  'blood',
  'butterfly',
  'cup',
  'dagger-duchess-tower-icon',
  'flower',
  'grave',
  'crown-icon',
  'ribbon',
  'ruin',
  'king-tower-level',
  'king-tower-ruin',
  'skull',
  'scoreboard',
  'snow',
  'circle',
]
object_unit_list = [
  'axe',
  'dirt',
  'goblin-ball',
  'bomb',
]
# Use in features building
bar2_unit_list = ['dagger-duchess-tower-bar', 'skeleton-king-bar']

if __name__ == '__main__':
  from katacr.utils import colorstr
  check_union = set(ground_unit_list).intersection(flying_unit_list)
  assert len(check_union) == 0, f"Ground and fly should no intersection element {check_union}."
  avail_units = set(ground_unit_list) | set(flying_unit_list) | set(other_unit_list) | set(tower_unit_list)
  # avail_units.remove('bar-level')
  unit_list.remove("selected")
  total_units = len(unit_list)
  print(colorstr(f"Total number unit n={total_units}"))
  # for i, u in enumerate(sorted(unit_list)):
  #   print(i+1, u, '✔' if u in avail_units else '✘')
  print(f"{colorstr(f'Available units (n={len(avail_units)})')}", sorted(avail_units))
  residue = set(unit_list) - avail_units
  print(f"{colorstr(f'Residue unit: (n={len(residue)})')}", sorted(residue))

  from katacr.build_dataset.constant import path_dataset
  path_segment = path_dataset / "images/segment"
  segment_units = {'0': set(), '1': set()}
  for pu in path_segment.glob('*'):
    if pu.name in ['backgrounds', 'background-items']:
      continue
    for pi in pu.glob('*.png'):
      name = pi.stem
      sl = name.split('_')
      segment_units[sl[1]].add(sl[0])
  # print(segment_units['0'])
  # print(segment_units['1'])
  # print(segment_units['0'] - segment_units['1'])
  residue_set = set(unit_list) - {'text', 'emote', 'evolution-symbol', 'dirt', 'elixir', 'axe', 'dagger-duchess-tower-bar'}
  residue_set_1 = residue_set - segment_units['1']
  print(f"{colorstr(f'Residue unit with 1 (n={len(residue_set_1)}, N-n={total_units-len(residue_set_1)})')}", sorted(residue_set_1))