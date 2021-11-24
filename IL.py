# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
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
import pandas as pd
from sklearn.model_selection import train_test_split


# %%
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

seed = 42
seed_everything(seed)


# %%
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
    else: # transfer
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
        label = None
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
    
    episodes = [path for path in Path(episode_dir).glob('episodes????????.json') if 'output' not in path.name]
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
    for filepath in episodes: 
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
        if int(submit_id) == json_load["agents"][0]["submissionId"] or int(submit_id) == json_load["agents"][1]["submissionId"]:
            episode_ids.append(json_load["id"])
        
    numofepi = len(episode_ids)
    print(f"submittion id {submit_id}. num of epi {numofepi}")
    return episode_ids


def create_dataset_from_submit_id(submit_id, episode_dir):
    obses = {}
    woker_samples = []
    citytile_samples = []

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


# %%
seed = 42
seed_everything(seed)

episode_dir = '/home/ubuntu/work/codes/imitation_learning/archive-2'

submit_ids = get_all_submit_ids(episode_dir)
max_scores = []
for submit_id in tqdm(submit_ids):
    max_score = get_max_score(submit_id, episode_dir)
    max_scores.append(max_score)
    # print(f"max score is {max_score} of {submit_id}")
max_scores_df = pd.DataFrame([submit_ids, max_scores]).T
max_scores_df.columns = ["submissionId", "max_score"]
max_scores_df = max_scores_df.sort_values("max_score",ascending=False)


# %%
obses1, woker_samples1, citytile_samples1 = create_dataset_from_submit_id("23297953", episode_dir)
obses2, woker_samples2, citytile_samples2 = create_dataset_from_submit_id("23281649", episode_dir)
obses3, woker_samples3, citytile_samples3 = create_dataset_from_submit_id("23032370", episode_dir)
obses4, woker_samples4, citytile_samples4 = create_dataset_from_submit_id("23554491", episode_dir)
obses5, woker_samples5, citytile_samples5 = create_dataset_from_submit_id("23554433", episode_dir)


# %%
print(len(obses1))
print(len(obses2))
print(len(obses3))
print(len(obses4))
print(len(obses5))


# %%
obses1.update(obses2)
obses1.update(obses3)
obses1.update(obses4)
obses1.update(obses5)


# %%
woker_samples1.extend(woker_samples2)
woker_samples1.extend(woker_samples3)
woker_samples1.extend(woker_samples4)
woker_samples1.extend(woker_samples5)


# %%
citytile_samples1.extend(citytile_samples2)
citytile_samples1.extend(citytile_samples3)
citytile_samples1.extend(citytile_samples4)
citytile_samples1.extend(citytile_samples5)


# %%
obses = obses1
woker_samples = woker_samples1
citytile_samples = citytile_samples1


# %%

print('obses:', len(obses), 'woker_samples:', len(woker_samples), 'citytile_samples:', len(citytile_samples))

woker_labels = [sample[-1] for sample in woker_samples]
actions = ['north', 'south', 'west', 'east', 'bcity']
for value, count in zip(*np.unique(woker_labels, return_counts=True)):
    print(f'{actions[value]:^5}: {count:>3}')

citytile_labels = [sample[-1] for sample in citytile_samples]
actions = ['research', 'build_worker', 'build_cart']
for value, count in zip(*np.unique(citytile_labels, return_counts=True)):
    print(f'{actions[value]:^5}: {count:>3}')


# %%
# build_cardのsampleは消しておく
citytile_samples = [citytile_sample for citytile_sample in citytile_samples if citytile_sample[2] != 2]
citytile_labels = [sample[-1] for sample in citytile_samples]
actions = ['research', 'build_worker', 'build_cart']
for value, count in zip(*np.unique(citytile_labels, return_counts=True)):
    print(f'{actions[value]:^5}: {count:>3}')
len(citytile_samples)


# %%
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


# %%
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


# %%
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


# %%


model = LuxNet_worker()
train, val = train_test_split(woker_samples, test_size=0.1, random_state=42, stratify=woker_labels)
batch_size = 64
train_loader = DataLoader(
    LuxDataset_worker(obses, train), 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)
val_loader = DataLoader(
    LuxDataset_worker(obses, val), 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)
dataloaders_dict = {"train": train_loader, "val": val_loader}
criterion = nn.CrossEntropyLoss()


# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3,model_type="worker")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3,model_type="worker")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3,model_type="worker")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3,model_type="worker")


# %%
model = LuxNet_citytile()
train, val = train_test_split(citytile_samples, test_size=0.1, random_state=42, stratify=citytile_labels)
batch_size = 64
train_loader = DataLoader(
    LuxDataset_citytile(obses, train), 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)
val_loader = DataLoader(
    LuxDataset_citytile(obses, val), 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)
dataloaders_dict = {"train": train_loader, "val": val_loader}
criterion = nn.CrossEntropyLoss()



# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3, model_type="citytile")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3, model_type="citytile")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3, model_type="citytile")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=3, model_type="citytile")


# %%
get_ipython().run_cell_magic('writefile', 'agent.py', 'import os\nimport numpy as np\nimport torch\nfrom lux.game import Game\nfrom torch import nn\nimport torch.nn.functional as F\n\n# Neural Network for Lux AI\nclass BasicConv2d(nn.Module):\n    def __init__(self, input_dim, output_dim, kernel_size, bn):\n        super().__init__()\n        self.conv = nn.Conv2d(\n            input_dim, output_dim, \n            kernel_size=kernel_size, \n            padding=(kernel_size[0] // 2, kernel_size[1] // 2)\n        )\n        self.bn = nn.BatchNorm2d(output_dim) if bn else None\n\n    def forward(self, x):\n        h = self.conv(x)\n        h = self.bn(h) if self.bn is not None else h\n        return h\n\n\nclass LuxNet_worker(nn.Module):\n    def __init__(self):\n        super().__init__()\n        layers, filters = 12, 32\n        input_layers = 20\n        action_num = 5\n        self.conv0 = BasicConv2d(input_layers, filters, (3, 3), True)\n        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])\n        self.head_p = nn.Linear(filters, action_num, bias=False)\n\n    def forward(self, x):\n        h = F.relu_(self.conv0(x))\n        for block in self.blocks:\n            h = F.relu_(h + block(h))\n        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)\n        p = self.head_p(h_head)\n        return p\n\nclass LuxNet_citytile(nn.Module):\n    def __init__(self):\n        super().__init__()\n        layers, filters = 12, 32\n        input_layers = 20\n        action_num = 2 # build_cart なし\n        self.conv0 = BasicConv2d(input_layers, filters, (3, 3), True)\n        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])\n        self.head_p = nn.Linear(filters, action_num, bias=False)\n\n    def forward(self, x):\n        h = F.relu_(self.conv0(x))\n        for block in self.blocks:\n            h = F.relu_(h + block(h))\n        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)\n        p = self.head_p(h_head)\n        return p\n\n# path = \'/kaggle_simulations/agent\' if os.path.exists(\'/kaggle_simulations\') else \'.\'\n# path = "."\n# worker_model = torch.jit.load(f\'{path}/worker_model.pth\')\n# worker_model.eval()\n\n# citytile_model = torch.jit.load(f\'{path}/citytile_model.pth\')\n# citytile_model.eval()\n\n\npath = \'/kaggle_simulations/agent\' if os.path.exists(\'/kaggle_simulations\') else \'.\'\ncitytile_model = LuxNet_citytile()\ncitytile_model.load_state_dict(torch.load(f\'{path}/citytile_state_dict\'))\ncitytile_model.eval()\n\nworker_model = LuxNet_worker()\nworker_model.load_state_dict(torch.load(f\'{path}/worker_state_dict\'))\nworker_model.eval()\n\n\n\ndef make_input_worker(obs, unit_id):\n    width, height = obs[\'width\'], obs[\'height\']\n    x_shift = (32 - width) // 2\n    y_shift = (32 - height) // 2\n    cities = {}\n    \n    b = np.zeros((20, 32, 32), dtype=np.float32)\n    \n    for update in obs[\'updates\']:\n        strs = update.split(\' \')\n        input_identifier = strs[0]\n        \n        if input_identifier == \'u\':\n            x = int(strs[4]) + x_shift\n            y = int(strs[5]) + y_shift\n            wood = int(strs[7])\n            coal = int(strs[8])\n            uranium = int(strs[9])\n            if unit_id == strs[3]:\n                # Position and Cargo\n                b[:2, x, y] = (\n                    1,\n                    (wood + coal + uranium) / 100\n                )\n            else:\n                # Units\n                team = int(strs[2])\n                cooldown = float(strs[6])\n                idx = 2 + (team - obs[\'player\']) % 2 * 3\n                b[idx:idx + 3, x, y] = (\n                    1,\n                    cooldown / 6,\n                    (wood + coal + uranium) / 100\n                )\n        elif input_identifier == \'ct\':\n            # CityTiles\n            team = int(strs[1])\n            city_id = strs[2]\n            x = int(strs[3]) + x_shift\n            y = int(strs[4]) + y_shift\n            idx = 8 + (team - obs[\'player\']) % 2 * 2\n            b[idx:idx + 2, x, y] = (\n                1,\n                cities[city_id]\n            )\n        elif input_identifier == \'r\':\n            # Resources\n            r_type = strs[1]\n            x = int(strs[2]) + x_shift\n            y = int(strs[3]) + y_shift\n            amt = int(float(strs[4]))\n            b[{\'wood\': 12, \'coal\': 13, \'uranium\': 14}[r_type], x, y] = amt / 800\n        elif input_identifier == \'rp\':\n            # Research Points\n            team = int(strs[1])\n            rp = int(strs[2])\n            b[15 + (team - obs[\'player\']) % 2, :] = min(rp, 200) / 200\n        elif input_identifier == \'c\':\n            # Cities\n            city_id = strs[2]\n            fuel = float(strs[3])\n            lightupkeep = float(strs[4])\n            cities[city_id] = min(fuel / lightupkeep, 10) / 10\n    \n    # Day/Night Cycle\n    b[17, :] = obs[\'step\'] % 40 / 40\n    # Turns\n    b[18, :] = obs[\'step\'] / 360\n    # Map Size\n    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1\n\n    return b\n\ndef make_input_citytile(obs, unit_id):\n    width, height = obs[\'width\'], obs[\'height\']\n    x_shift = (32 - width) // 2\n    y_shift = (32 - height) // 2\n    cities = {}\n    \n    b = np.zeros((20, 32, 32), dtype=np.float32)\n    \n    for update in obs[\'updates\']:\n        strs = update.split(\' \')\n        input_identifier = strs[0]\n        \n        if input_identifier == \'u\':\n            x = int(strs[4]) + x_shift\n            y = int(strs[5]) + y_shift\n            wood = int(strs[7])\n            coal = int(strs[8])\n            uranium = int(strs[9])\n            # Units\n            team = int(strs[2])\n            cooldown = float(strs[6])\n            idx = 2 + (team - obs[\'player\']) % 2 * 3\n            b[idx:idx + 3, x, y] = (\n                1,\n                cooldown / 6,\n                (wood + coal + uranium) / 100\n            )\n        elif input_identifier == \'ct\':\n            # CityTiles\n            citytile_pos = unit_id.split(\' \')\n\n            if citytile_pos[0] == strs[3] and citytile_pos[1] == strs[4]:  #\u3000自分自身なら\n                x = int(strs[3]) + x_shift\n                y = int(strs[4]) + y_shift\n                b[:2, x, y] = (\n                    1,\n                    cities[city_id]\n                )\n            else:\n                team = int(strs[1])\n                city_id = strs[2]\n                x = int(strs[3]) + x_shift\n                y = int(strs[4]) + y_shift\n                idx = 8 + (team - obs[\'player\']) % 2 * 2\n                b[idx:idx + 2, x, y] = (\n                    1,\n                    cities[city_id]\n                )\n        elif input_identifier == \'r\':\n            # Resources\n            r_type = strs[1]\n            x = int(strs[2]) + x_shift\n            y = int(strs[3]) + y_shift\n            amt = int(float(strs[4]))\n            b[{\'wood\': 12, \'coal\': 13, \'uranium\': 14}[r_type], x, y] = amt / 800\n        elif input_identifier == \'rp\':\n            # Research Points\n            team = int(strs[1])\n            rp = int(strs[2])\n            b[15 + (team - obs[\'player\']) % 2, :] = min(rp, 200) / 200\n        elif input_identifier == \'c\':\n            # Cities\n            city_id = strs[2]\n            fuel = float(strs[3])\n            lightupkeep = float(strs[4])\n            cities[city_id] = min(fuel / lightupkeep, 10) / 10\n    \n    # Day/Night Cycle\n    b[17, :] = obs[\'step\'] % 40 / 40\n    # Turns\n    b[18, :] = obs[\'step\'] / 360\n    # Map Size\n    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1\n\n    return b\n\n\ngame_state = None\ndef get_game_state(observation):\n    global game_state\n    \n    if observation["step"] == 0:\n        game_state = Game()\n        game_state._initialize(observation["updates"])\n        game_state._update(observation["updates"][2:])\n        game_state.id = observation["player"]\n    else:\n        game_state._update(observation["updates"])\n    return game_state\n\n\ndef in_city(pos):    \n    try:\n        city = game_state.map.get_cell_by_pos(pos).citytile\n        return city is not None and city.team == game_state.id\n    except:\n        return False\n\n\ndef call_func(obj, method, args=[]):\n    return getattr(obj, method)(*args)\n\n\nunit_actions = [(\'move\', \'n\'), (\'move\', \'s\'), (\'move\', \'w\'), (\'move\', \'e\'), (\'build_city\',)]\ndef get_action_worker(policy, unit, dest):\n    for label in np.argsort(policy)[::-1]:\n        act = unit_actions[label]\n        pos = unit.pos.translate(act[-1], 1) or unit.pos # 移動できたかそのままか\n        if pos not in dest or in_city(pos):\n            return call_func(unit, *act), pos \n        \n    return unit.move(\'c\'), unit.pos\n\n\ncity_actions = [(\'research\',), (\'build_worker\',)] \ndef get_action_citytile(policy, city_tile):\n    for label in np.argsort(policy)[::-1]:\n        act = city_actions[label]\n        return call_func(city_tile, *act), None\n\n\ndef agent(observation, configuration):\n    global game_state\n    \n    game_state = get_game_state(observation)    \n    player = game_state.players[observation.player]\n    actions = []\n    \n    # City Actions\n    for city in player.cities.values():\n        for city_tile in city.citytiles:\n            if city_tile.can_act():\n                if not player.researched_uranium():\n                    citytile_pos = f"{city_tile.pos.x} {city_tile.pos.y}"\n                    state = make_input_citytile(observation, citytile_pos)\n                    with torch.no_grad():\n                        p = citytile_model(torch.from_numpy(state).unsqueeze(0))\n\n                    policy = p.squeeze(0).numpy()\n\n                    action, _ = get_action_citytile(policy, city_tile=city_tile)\n                else: # research完了してたらbuildworker\n                    action = city_tile.build_worker()\n                actions.append(action)\n    \n    # Worker Actions\n    dest = []\n    for unit in player.units:\n        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):\n            state = make_input_worker(observation, unit.id)\n            with torch.no_grad():\n                p = worker_model(torch.from_numpy(state).unsqueeze(0))\n\n            policy = p.squeeze(0).numpy()\n\n            action, pos = get_action_worker(policy, unit=unit, dest=dest)\n            actions.append(action)\n            dest.append(pos)\n\n    return actions\n')


# %%
from kaggle_environments import make

env = make("lux_ai_2021", configuration={"width": 24, "height": 24, "loglevel": 2, "annotations": True}, debug=False)
steps = env.run(['agent.py', 'agent.py'])


# %%
get_ipython().system('tar -czf submission.tar.gz *')


# %%



