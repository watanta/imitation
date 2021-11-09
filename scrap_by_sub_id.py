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

    return obses, woker_samples, citytile_samples

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
        input_layers = 20
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
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:

            traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32)) # state_shape 何これ？
            traced.save(f'{model_type}_model.pth')
            torch.save(model.state_dict(), f'{model_type}_state_dict')
            best_acc = epoch_acc




if __name__ == "__main__":

    seed = 42
    seed_everything(seed)

    episode_dir = '/home/ubuntu/work/codes/imitation_learning/input/episodes'
    obses, woker_samples, citytile_samples = create_dataset_from_json(episode_dir)
    print('obses:', len(obses), 'woker_samples:', len(woker_samples), 'citytile_samples:', len(citytile_samples))

    woker_labels = [sample[-1] for sample in woker_samples]
    actions = ['north', 'south', 'west', 'east', 'bcity']
    for value, count in zip(*np.unique(woker_labels, return_counts=True)):
        print(f'{actions[value]:^5}: {count:>3}')

    citytile_labels = [sample[-1] for sample in citytile_samples]
    actions = ['research', 'build_worker', 'build_cart']
    for value, count in zip(*np.unique(citytile_labels, return_counts=True)):
        print(f'{actions[value]:^5}: {count:>3}')

# model = LuxNet_worker()
# train, val = train_test_split(woker_samples, test_size=0.1, random_state=42, stratify=woker_labels)
# batch_size = 64
# train_loader = DataLoader(
#     LuxDataset_worker(obses, train), 
#     batch_size=batch_size, 
#     shuffle=True, 
#     num_workers=2
# )
# val_loader = DataLoader(
#     LuxDataset_worker(obses, val), 
#     batch_size=batch_size, 
#     shuffle=False, 
#     num_workers=2
# )
# dataloaders_dict = {"train": train_loader, "val": val_loader}
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=1, model_type="worker")

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
# env.render(mode="ipython", width=1200, height=800)

pass
