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
  'bowl',
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
  'archer-evolution'
  # 2023/11/29: add
  'ice-spirit-evolution',
  # 2024/1/1: add
  'cannoneer-tower',
  'valkyrie-evolution',
]

idx2unit = dict(enumerate(unit_list))
unit2idx = {name: idx for idx, name in enumerate(unit_list)}

ground_unit_list = [
  'barbarian',
  'barbarian-barrel',
  'barbarian-hut',
  'cannon',
  'electro-spirit',
  'goblin-brawler',
  'goblin-cage',
  'hog-rider',
  'ice-golem',
  'ice-spirit',
  'musketeer',
  'royal-hog',
  'royal-recruit',
  'skeleton',
  'the-log',
  'zappy',
]

tower_unit_list = [
  'king-tower',
  'queen-tower',
]

flying_unit_list = [
  'flying-machine',
  'fireball',
  'arrows',
]
spell_unit_list = [
  'fireball',
  'arrows',
]

other_unit_list = [
  'bar',
  'tower-bar',
  'king-tower-bar',
  'clock',
  'text',
  'elixir',
  'emote',
]