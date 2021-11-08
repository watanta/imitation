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


def to_label(action):
    strs = action.split(' ')
    if strs[0] == "m" or strs[0] == "bcity": # if worker or cart
        unit_id = strs[1]
        if strs[0] == 'm':
            label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
        elif strs[0] == 'bcity':
            label = 4
    elif strs[0] == "bw" or strs[0] == "bc" or strs[0] == "r": # if citytile
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



def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='Toad Brigade'): 
    obses = {}
    samples = []
    append = samples.append
    
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
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
                    unit_id, label = to_label(action)
                    if label is not None: # move c or transfer 
                        append((obs_id, unit_id, label))

    return obses, samples


if __name__ == "__main__":


# obs = np.zeros((32,20,32,32))
# actions = np.ones((32))
# next_obs = np.zeros((32,20,32,32))
# dones = np.array([True]*32)
# infos = np.array([{}]*32)
# cat_parts = {"obs":obs, "acts":actions, "next_obs":next_obs, "dones":dones, "infos":infos}
# transitions = types.Transitions(**cat_parts)

    seed = 42
    seed_everything(seed)

    episode_dir = '/home/ubuntu/work/codes/imitation_learning/input/episodes'
    obses, samples = create_dataset_from_json(episode_dir)
    print('obses:', len(obses), 'samples:', len(samples))

    labels = [sample[-1] for sample in samples]
    actions = ['north', 'south', 'west', 'east', 'bcity']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        print(f'{actions[value]:^5}: {count:>3}')

    trainsitions = []
    unique_episodes = get_unique_episodes(obses)
    for episode in unique_episodes:
        obs_in_episode = get_obs_in_epidode(obses)
        for obs in obses:
            # BCではobsとactsだけ使う
            transition = {"obs":None, "acts":None, "next_obs":None, "dones":None, "infos":None} 
            transition["obs"] = obs
            transition["acts"] = get_sample_of_obs(obs)[2]
            transition["dones"] = is_done_obs(ons)
            if trainsition["dones"] = True:
                transition["next_obs"] = None
            else:
                trainsition["next_obs"] = next(obs)
            transition["infos"] = None




