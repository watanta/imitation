import sys
print(sys.version)

import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn import Conv2d
from functools import reduce
from operator import add
import datetime

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def to_label_for_citytile(action):
    strs = action.split(' ')
    if strs[0] == "bw" or strs[0] == "bc" or strs[0] == "r": # if citytile
        unit_id = strs[1]+" "+strs[2] # citytileはidを持たないので、位置を入れて自分を判別
        if strs[0] == "r":
            label = 0
        elif strs[0] == "bw":
            label = 1
        elif strs[0] == "bc":
            label = 2
    else: 
        unit_id = None
        label = None
    return unit_id, label

def to_label_for_worker(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    else:
        label = None #transfer
    return unit_id, label


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True

def action_is_worker(action):
    if "u" in action: #TODO ほんとにこれでいいか
        return True
    else:
        return False

def action_is_citytile(action):
    if ("r" in action) or ("bw" in action) or ("bc" in action):
        return True
    else:
        return False


def create_dataset_from_json(episode_dir, team_name='Toad Brigade'): 
    obses = {}
    woker_samples = []
    citytile_samples = []
    action_samples = []
    
    episodes = [path for path in Path(episode_dir).glob('????????.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs
                action_samples = [(obs_id, actions)]
                                
                for action in actions:
                    if action_is_worker(action):
                        unit_id, label = to_label_for_worker(action)
                        if label is not None: #transfer落とす
                            woker_samples.append((obs_id, unit_id, label))
                    elif action_is_citytile(action): 
                         unit_id, label = to_label_for_citytile(action)
                         citytile_samples.append((obs_id, unit_id, label))
                    else: #cart
                        unit_id = None
                        label = None

    return obses, woker_samples, citytile_samples, action_samples

# Input for Neural Network
def make_input_worker(obs, unit_id):
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
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
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

class LuxDataset_UNet(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        actions = sample[1]
        obs_id = sample[0]
        obs = self.obses[obs_id]
        state, action_map = make_input_for_UNet(obs, actions)
        
        return state, action_map


class LuxDataset_worker(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input_worker(obs, unit_id)
        
        return state, action

class LuxDataset_citytile(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input_citytile(obs, unit_id)
        
        return state, action

# Neural Network for Lux AI
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


class LuxNet_worker(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        input_layers = 18
        action_num = 5
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

class LuxNet_citytile(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        input_layers = 20
        action_num = 2 # buil_cart なし
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

def get_acc(preds, actions):
    batch_num = preds.shape[0]
    map_h = preds.shape[1]
    map_w = preds.shape[2]
    label = torch.max(actions, 1)[1]
    losses = []
    pred_num = 0
    TP_num = 0

    TP_mat = preds == label
    unit_exist = torch.sum(actions, 1) == 1
    TP_num = (unit_exist * TP_mat).sum() # workerが存在する位置のみacc計算する

    acc = TP_num / unit_exist.sum()

    return acc


def criterion_UNet(policy, actions):
    batch_num = policy.shape[0]
    map_h = policy.shape[2]
    map_w = policy.shape[3]
    losses = []
    for batch in range(batch_num):
        for y in range(map_h):
            for x in range(map_w):
                if actions[batch, :, x, y].sum() == 1: # workerが存在する位置のみloss計算する
                    p = torch.reshape(policy[batch, :, x, y], (1, -1))
                    l = torch.argmax(actions[batch, :, x, y]).reshape([1])
                    loss = nn.CrossEntropyLoss()(p, l)
                    losses.append(loss)
    
    loss = reduce(add, losses)
    return loss / len(losses)
                    

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, model_type):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion_UNet(policy, actions)
                    # loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    # epoch_acc += torch.sum(preds == actions.data)
                    epoch_acc += get_acc(preds, actions)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / len(dataloader)
            epoch_acc = epoch_acc / len(dataloader)

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:

            # traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32)) # state_shape 何これ？
            # traced.save(f'{model_type}_model.pth')
            torch.save(model.state_dict(), f'{model_type}_state_dict')
            best_acc = epoch_acc

def get_all_submit_ids(episode_dir):
    submit_id_set = set([])
    episodes = [path for path in Path(episode_dir).glob('????????_info.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)
        player0 = json_load["agents"][0]["submissionId"]
        player1 = json_load["agents"][0]["submissionId"]
        submit_id_set.add(player0)
        submit_id_set.add(player1)
    return submit_id_set

def get_max_score(submit_id, episode_dir):
    max_score = -1000
    episodes = [path for path in Path(episode_dir).glob('????????_info.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)
        player0 = json_load["agents"][0]["submissionId"]
        player1 = json_load["agents"][1]["submissionId"]
        if player0 == submit_id:
            max_score = max(max_score, json_load["agents"][0]["updatedScore"])
        elif player1 == submit_id:
            max_score = max(max_score, json_load["agents"][1]["updatedScore"])
        else:
            pass
    return max_score


def get_all_episode_ids(submit_id, episode_dir):
    episode_ids = []
    episodes = [path for path in Path(episode_dir).glob('????????_info.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)
        if int(submit_id) == json_load["agents"][0]["submissionId"]:
            episode_ids.append(json_load["id"])
    return episode_ids


def create_dataset_from_submit_id(submit_id, episode_dir):
    obses = {}
    woker_samples = []
    citytile_samples = []
    action_samples = []

    episode_ids = get_all_episode_ids(submit_id, episode_dir)
    episodes = [episode_dir+"/"+str(episode_id)+".json" for episode_id in episode_ids]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs
                action_samples.append((obs_id, actions))

                for action in actions:
                    if action_is_worker(action):
                        unit_id, label = to_label_for_worker(action)
                        if label is not None: #transfer落とす
                            woker_samples.append((obs_id, unit_id, label))
                    elif action_is_citytile(action): 
                         unit_id, label = to_label_for_citytile(action)
                         citytile_samples.append((obs_id, unit_id, label))
                    else: #cart
                        unit_id = None
                        label = None

    return obses, woker_samples, citytile_samples, action_samples

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


if __name__ == "__main__":

    seed = 42
    seed_everything(seed)

    print(f"start {datetime.datetime.now()}")

    episode_dir = '/home/ubuntu/work/codes/imitation_learning/archive-2'
    # submit_ids = get_all_submit_ids(episode_dir)
    # max_scores = []
    # for submit_id in submit_ids:
    #     max_score = get_max_score(submit_id, episode_dir)
    #     max_scores.append(max_score)
    #     # print(f"max score is {max_score} of {submit_id}")
    # max_scores_df = pd.DataFrame([submit_ids, max_scores]).T
    # max_scores_df.columns = ["submissionId", "max_score"]
    # max_scores_df = max_scores_df.sort_values("max_score",ascending=False)



    # obses1, woker_samples1, citytile_samples1, action_samples1 = create_dataset_from_submit_id("23297953", episode_dir)
    # obses2, woker_samples2, citytile_samples2, action_samples2 = create_dataset_from_submit_id("23281649", episode_dir)
    # obses3, woker_samples3, citytile_samples3, action_samples3 = create_dataset_from_submit_id("23032370", episode_dir)

    # obses1.update(obses2)
    # obses1.update(obses3)

    # action_samples1.extend(action_samples2)
    # action_samples1.extend(action_samples3)

    # obses = obses1
    # action_samples = action_samples1


    # model = LuxNet_worker()
    # n_channels = 18
    # n_classes = 6
    # model = UNet_worker(n_channels, n_classes, bilinear=False)
    # train, val = train_test_split(action_samples, test_size=0.1, random_state=42)
    # th = 1000
    # train = random.sample(train, th) # 時間がかかるのでサンプル数絞っておく
    # val = random.sample(val, th) # 時間がかかるのでサンプル数絞っておく
    # batch_size = 128
    # train_loader = DataLoader(
    #     LuxDataset_UNet(obses, train), 
    #     batch_size=batch_size, 
    #     shuffle=True, 
    #     num_workers=2
    # )
    # val_loader = DataLoader(
    #     LuxDataset_UNet(obses, val), 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=2
    # )
    # dataloaders_dict = {"train": train_loader, "val": val_loader}
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10, model_type="worker")

    # print(f"end {datetime.datetime.now()}")

    # model = LuxNet_citytile()
    # train, val = train_test_split(citytile_samples, test_size=0.1, random_state=42, stratify=citytile_labels)
    # batch_size = 64
    # train_loader = DataLoader(
    #     LuxDataset_citytile(obses, train), 
    #     batch_size=batch_size, 
    #     shuffle=True, 
    #     num_workers=2
    # )
    # val_loader = DataLoader(
    #     LuxDataset_citytile(obses, val), 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=2
    # )
    # dataloaders_dict = {"train": train_loader, "val": val_loader}
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=1, model_type="citytile")

    from kaggle_environments import make

    env = make("lux_ai_2021", configuration={"width": 24, "height": 24, "loglevel": 2, "annotations": True}, debug=False)
    steps = env.run(['agent.py', 'agent.py'])
    env.render(mode="ipython", width=1200, height=800)
    pass
