import os
import numpy as np
import torch
from lux.game import Game
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_worker(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_worker, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(256+256, 256 // factor, bilinear)
        self.up3 = Up(256+128, 128 // factor, bilinear)
        self.up4 = Up(128+64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet_citytile(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        input_layers = 20
        action_num = 2 # build_cart なし
        self.conv0 = BasicConv2d(input_layers, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, action_num, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p

# path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
# path = "."
# worker_model = torch.jit.load(f'{path}/worker_model.pth')
# worker_model.eval()

# citytile_model = torch.jit.load(f'{path}/citytile_model.pth')
# citytile_model.eval()


path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
citytile_model = LuxNet_citytile()
citytile_model.load_state_dict(torch.load(f'{path}/citytile_state_dict'))
citytile_model.eval()

n_channels = 18
n_classes = 6
worker_model = UNet_worker(n_channels, n_classes, bilinear=False)
worker_model.load_state_dict(torch.load(f'{path}/worker_state_dict'))
worker_model.eval()


def get_unit_position_dict(obs):
    unit_position_dict = {}
    for update in obs["updates"]:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == "u":
            unit_id = strs[3]
            unit_position_dict[unit_id] = (int(strs[4]), int(strs[5]))
    return unit_position_dict


def make_input_for_UNet(obs, actions):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((18, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            # Units
            team = int(strs[2])
            cooldown = float(strs[6])
            idx = (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100
            )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 6 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 10, 'coal': 11, 'uranium': 12}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[13 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[15, :] = obs['step'] % 40 / 40
    # Turns
    b[16, :] = obs['step'] / 360
    # Map Size
    b[17, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    # make action_map
    action_map = np.zeros((6, 32, 32), dtype=np.float32) #action_num x map_h x map_w

    unit_positions_dict = get_unit_position_dict(obs)

    for action in actions:
        strs = action.split(' ')
        
        if strs[0] == 'm':
            unit_id = strs[1]
            action = strs[2]
            pos_x, pos_y = unit_positions_dict[unit_id]
            if action == "n":
                label = 0
            elif action == "s":
                label = 1
            elif action == "w":
                label = 2
            elif action == "e":
                label = 3
            elif action == "c":
                label = 4
            action_map[label][pos_x][pos_y] = 1
        elif strs[0] == "bcity":
            unit_id = strs[1]
            label = 5
            pos_x, pos_y = unit_positions_dict[unit_id]        
            action_map[label][pos_x][pos_y] = 1

    return b, action_map


def make_input_citytile(obs, unit_id):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            # Units
            team = int(strs[2])
            cooldown = float(strs[6])
            idx = 2 + (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100
            )
        elif input_identifier == 'ct':
            # CityTiles
            citytile_pos = unit_id.split(' ')

            if citytile_pos[0] == strs[3] and citytile_pos[1] == strs[4]:  #　自分自身なら
                x = int(strs[3]) + x_shift
                y = int(strs[4]) + y_shift
                b[:2, x, y] = (
                    1,
                    cities[city_id]
                )
            else:
                team = int(strs[1])
                city_id = strs[2]
                x = int(strs[3]) + x_shift
                y = int(strs[4]) + y_shift
                idx = 8 + (team - obs['player']) % 2 * 2
                b[idx:idx + 2, x, y] = (
                    1,
                    cities[city_id]
                )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    # Map Size
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


game_state = None
def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state._update(observation["updates"])
    return game_state


def in_city(pos):    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


unit_actions = [(('move', 'c')), ('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
def get_action_worker(policy, unit, dest):
    for label in np.argsort(policy[:, unit.pos.x, unit.pos.y])[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos # 移動できたかそのままか
        if pos not in dest or in_city(pos):
            return call_func(unit, *act), pos 
        
    return unit.move('c'), unit.pos


city_actions = [('research',), ('build_worker',)] 
def get_action_citytile(policy, city_tile):
    for label in np.argsort(policy)[::-1]:
        act = city_actions[label]
        return call_func(city_tile, *act), None


def agent(observation, configuration):
    global game_state
    
    game_state = get_game_state(observation)    
    player = game_state.players[observation.player]
    actions = []
    
    # City Actions
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                if not player.researched_uranium():
                    citytile_pos = f"{city_tile.pos.x} {city_tile.pos.y}"
                    state = make_input_citytile(observation, citytile_pos)
                    with torch.no_grad():
                        p = citytile_model(torch.from_numpy(state).unsqueeze(0))

                    policy = p.squeeze(0).numpy()

                    action, _ = get_action_citytile(policy, city_tile=city_tile)
                else: # research完了してたらbuildworker
                    action = city_tile.build_worker()
                actions.append(action)
    
    # Worker Actions
    dest = []
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            state, _ = make_input_for_UNet(observation, unit.id)
            with torch.no_grad():
                p = worker_model(torch.from_numpy(state).unsqueeze(0))

            policy = p.squeeze(0).numpy()

            action, pos = get_action_worker(policy, unit=unit, dest=dest)
            actions.append(action)
            dest.append(pos)

    return actions
