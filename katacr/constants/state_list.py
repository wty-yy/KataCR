# -*- coding: utf-8 -*-
'''
@File    : state_list.py
@Time    : 2023/11/11 10:49:01
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.xyz/
@Desc    : 
Based on `Clash Royale dataset annotation.md`, we define all the state with special ids.

`K` means the classification dimension for each `class_name`.
| class_name    | K |
|---------------|---|
| belong        | 2 |
| movement      | 5 |
| shield/change | 2 |
| visible       | 2 |
| rage          | 2 |
| slow          | 2 |
| heal/clone    | 3 |

# Need predict:
blong:				0/1  # firend/enermy after card name
movement:			norm(walk/wait)/attack/deploy/freeze/dash(destory)
shield or change:	bare(charge)/shield(over)
visible:			visible/invisible
rage:				  norm/rage
slow:				  norm/slow
heal or clone:  norm/heal/clone

# Constants:
height:				ground/air
object:				unit/object
'''

state2idx = {  # key: (class, id)
  '0': (0, 0),  # Class 1: Belong (binary: 1 variable)
  '1': (0, 1),
  'attack': (1, 1),  # Class 2: Movement (mutli classes: 5 variables, default 0)
  'deploy': (1, 2),
  'freeze': (1, 3),
  'dash':   (1, 4),
  'destory':(1, 4),
  'dash|destory':(1, 4),
  'charge': (2, 0),  # Class 3: Shield/Charge (binary: 1 variable)
  'bare|charge': (2, 0),  # Class 3: Shield/Charge (binary: 1 variable)
  'shield': (2, 1),
  'over':   (2, 1),
  'shield|over': (2, 1),
  'invisible': (3, 1),  # Class 4: Visible (binary: 1 variable)
  'rage':  (4, 1),  # Class 5: Rage (binary: 1 variable)
  'slow':  (5, 1),  # Class 6: Slow (binary: 1 variable)
  'heal':  (6, 1),  # Class 7: Heal/Clone (multi classes: 3 variables)
  'clone': (6, 2),
}
# num_state_classes = 1 * 5 + 5 + 3  # 13
num_state_classes = 1  # just consider side
idx2state = {
  0: '0',
  1: '1',
  11: 'attack',
  12: 'deploy',
  13: 'freeze',
  14: 'dash|destory',
  20: 'bare|charge',
  21: 'shield|over',
  31: 'invisible',
  41: 'rage',
  51: 'slow',
  61: 'heal',
  62: 'clone',
}

unit2height = {  # (7, gound/air)
  'queen-tower': 0,
  'king-tower': 0,
  'skeleton': 0,
  'skeleton-evolution': 0,
  'electro-spirit': 0,
  'fire-spirit': 0,
  'ice-spirit': 0,
  'heal-spirit': 0,
  'goblin': 0,
  'spear-goblin': 0,
  'bomber': 0,
  'bat': 1,
  'bat-evolution': 1,
  'zap': 1,
  'giant-snowball': 1,
  'ice-golem': 0,
  'barbarian-barrel': 0,
  'barbarian': 0,
  'barbarian-evolution': 0,
  'wall-breaker': 0,
  'rage': 1,
  'the-log': 0,
  'archer': 0,
  'arrows': 1,
  'knight': 0,
  'knight-evolution': 0,
  'minion': 1,
  'cannon': 0,
  'skeleton-barrel': 1,
  'firecracker': 0,
  'firecracker-evolution': 0,
  'royal-delivery': 1,
  'royal-recruit': 0,
  'royal-recruit-evolution': 0,
  'tombstone': 0,
  'mega-minion': 1,
  'dart-goblin': 0,
  'earthquake': 0,
  'elixir-golem-big': 0,
  'elixir-golem-mid': 0,
  'elixir-golem-small': 0,
  'goblin-barrel': 1,
  'guard': 0,
  'clone': 1,
  'tornado': 1,
  'miner': 0,
  'dirt': 0,
  'princess': 0,
  'ice-wizard': 0,
  'royal-ghost': 0,
  'bandit': 0,
  'fisherman': 0,
  'skeleton-dragon': 1,
  'mortar': 0,
  'mortar-evolution': 0,
  'tesla': 0,
  'fireball': 1,
  'mini-pekka': 0,
  'musketeer': 0,
  'goblin-cage': 0,
  'goblin-brawler': 0,
  'valkyrie': 0,
  'battle-ram': 0,
  'bomb-tower': 0,
  'bomb': 0,
  'flying-machine': 1,
  'hog-rider': 0,
  'battle-healer': 0,
  'furnace': 0,
  'zappy': 0,
  'baby-dragon': 1,
  'dark-prince': 0,
  'freeze': 1,
  'poison': 1,
  'hunter': 0,
  'goblin-drill': 0,
  'electro-wizard': 0,
  'inferno-dragon': 1,
  'phoenix-big': 1,
  'phoenix-egg': 0,
  'phoenix-small': 1,
  'magic-archer': 0,
  'lumberjack': 0,
  'night-witch': 0,
  'mother-witch': 0,
  'hog': 0,
  'golden-knight': 0,
  'skeleton-king': 0,
  'mighty-miner': 0,
  'rascal-boy': 0,
  'rascal-girl': 0,
  'giant': 0,
  'goblin-hut': 0,
  'inferno-tower': 0,
  'wizard': 0,
  'royal-hog': 0,
  'witch': 0,
  'balloon': 1,
  'prince': 0,
  'electro-dragon': 1,
  'bowler': 0,
  'bowl': 0,
  'executioner': 0,
  'axe': 1,
  'cannon-cart': 0,
  'ram-rider': 0,
  'graveyard': 0,
  'archer-queen': 0,
  'monk': 0,
  'royal-giant': 0,
  'royal-giant-evolution': 0,
  'elite-barbarian': 0,
  'rocket': 1,
  'barbarian-hut': 0,
  'elixir-collector': 0,
  'giant-skeleton': 0,
  'lightning': 1,
  'goblin-giant': 0,
  'x-bow': 0,
  'sparky': 0,
  'pekka': 0,
  'electro-giant': 0,
  'mega-knight': 0,
  'lava-hound': 1,
  'lava-pup': 1,
  'golem': 0,
  'golemite': 0,
}

is_object = [
  # spells
  'zap',
  'giant-snowball',
  'rage',
  'the-log',
  'arrows',
  'earthquake',
  'clone',
  'tornado',
  'fireball',
  'freeze',
  'poison',
  'rocket',
  'lightning',
  # others
  'dirt',  # miner, goblin-drill, mighty-miner
  'bowl',  # bowler
  'axe',  # executioner
]
