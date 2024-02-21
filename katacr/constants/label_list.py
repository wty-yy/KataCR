"""
2023/11/01: Total unit numbers: 125 + 9 = 134
2023/11/11: Add little-prince, royal-guradian and archer-evolution, total: 137
"""
unit_list = [
  # object
  'king-tower',
  'queen-tower',
  'tower-bar',
  'bar',
  'king-tower-bar',
  'selected',
  'clock',
  'emote',
  'text',
  'elixir',
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
  # 'axe',
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
  'cannoneer-tower',
  'valkyrie-evolution',
  # 2024/2/1: add
  'bomber-evolution',
  'wall-breaker-evolution',
  # 2024/2/19: add
  'evolution-symbol'
]

idx2unit = dict(enumerate(unit_list))
unit2idx = {name: idx for idx, name in enumerate(unit_list)}

ground_unit_list = [
  'archer',
  'bandit',
  'barbarian',
  'barbarian-barrel',
  'barbarian-hut',
  'bomb',
  'bomb-tower',
  'cannon',
  'electro-spirit',
  'electro-wizard',
  'elite-barbarian',
  'firecracker',
  'firecracker-evolution',
  'goblin',
  'goblin-brawler',
  'goblin-cage',
  'golem',
  'golemite',
  'hog-rider',
  'ice-golem',
  'ice-spirit',
  'inferno-tower',
  'knight',
  'little-prince',
  'lumberjack',
  'magic-archer',
  'mighty-miner',
  'musketeer',
  'pekka',
  'poison',
  'princess',
  'rage',
  'ram-rider',
  'royal-delivery',
  'royal-ghost',
  'royal-giant',
  'royal-giant-evolution',
  'royal-guardian',
  'royal-hog',
  'royal-recruit',
  'skeleton',
  'skeleton-evolution',
  'tesla',
  'the-log',
  'tornado',
  'x-bow',
  'zappy',
]

tower_unit_list = [
  'king-tower',
  'queen-tower',
]

flying_unit_list = [
  'arrows',
  'baby-dragon',
  'flying-machine',
  'fireball',
  'goblin-barrel',
  'inferno-dragon',
  'mega-minion',
  'minion',
  'rocket',
  'zap',
]
spell_unit_list = [
  'arrows',
  'fireball',
  'goblin-barrel',
  'poison',
  'rage',
  'rocket',
  'tornado',
  'zap',
]

other_unit_list = [
  'bar',
  'bar-level',
  'evolution-symbol',
  'tower-bar',
  'king-tower-bar',
  'clock',
  'text',
  'elixir',
  'emote',
]

background_item_list = [
  'blood',
  'butterfly',
  'flower',
  'ribbon',
  'skull',
  'cup',
  'snow'
]

if __name__ == '__main__':
  check_union = set(ground_unit_list).intersection(flying_unit_list)
  assert len(check_union) == 0, f"Ground and fly should no intersection element {check_union}."
  avail_units = set(ground_unit_list) | set(flying_unit_list) | set(other_unit_list) | set(tower_unit_list)
  avail_units.remove('bar-level')
  print(f"Total number unit: ({len(unit_list)})")
  unit_list.remove("selected")
  for i, u in enumerate(sorted(unit_list)):
    print(i+1, u, '✔' if u in avail_units else '✘')
  print(f"Available units ({len(avail_units)}):", sorted(avail_units))
  residue = set(unit_list) - avail_units
  print(f"Residue unit: ({len(residue)})", sorted(residue))