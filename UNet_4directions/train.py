import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm import tqdm
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


seed = 42
seed_everything(seed)


def to_label(action, obs):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'n': 0, 'w': 1, 's': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    elif strs[0] == 't':
        label = 5   # CENTER (or transfer action if we want to perform better (to be implemented) )
    else:
        label = None

    unit_pos = (0, 0)

    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    for update in obs["updates"]:
        strs = update.split(" ")
        if strs[0] == "u" and strs[3] == unit_id:
            unit_pos = (int(strs[4]) + x_shift, int(strs[5]) + y_shift)
    return unit_id, label, unit_pos


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

    # if you want to use a small dataset, you can use the code at the next line
    # episodes = episodes[-500:]

    for filepath in tqdm(episodes):
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        for i in range(len(json_load['steps']) - 1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i + 1][index]['action']
                obs = json_load['steps'][i][0]['observation']

                if depleted_resources(obs):
                    break

                obs['player'] = index
                obs = dict([
                    (k, v) for k, v in obs.items()
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])

                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs


                # action_map for FOUR directions
                action_map_n = np.zeros((32, 32)) - 1
                action_map_w = np.zeros((32, 32)) - 1
                action_map_s = np.zeros((32, 32)) - 1
                action_map_e = np.zeros((32, 32)) - 1

                for action in actions:
                    unit_id, label, unit_pos = to_label(action, obs)
                    if label == 0:      # north
                        action_map_n[unit_pos[1], unit_pos[0]] = 0
                    elif label == 1:    # west
                        action_map_e[unit_pos[1], unit_pos[0]] = 0
                    elif label == 2:    # south
                        action_map_s[unit_pos[1], unit_pos[0]] = 0
                    elif label == 3:    # east
                        action_map_w[unit_pos[1], unit_pos[0]] = 0
                    elif label == 4:    # build_city
                        action_map_n[unit_pos[1], unit_pos[0]] = 1
                        action_map_w[unit_pos[1], unit_pos[0]] = 1
                        action_map_s[unit_pos[1], unit_pos[0]] = 1
                        action_map_e[unit_pos[1], unit_pos[0]] = 1
                    elif label == 5:    # transfer(or CENTER)
                        action_map_n[unit_pos[1], unit_pos[0]] = 2
                        action_map_w[unit_pos[1], unit_pos[0]] = 2
                        action_map_s[unit_pos[1], unit_pos[0]] = 2
                        action_map_e[unit_pos[1], unit_pos[0]] = 2
                # we only take the training data with workers' actions
                if np.any(action_map_n+1):
                    # the 3rd number:0,1,2,3 means the time we should rotate our map
                    append((obs_id, action_map_n, 0))
                if np.any(action_map_w+1):
                    append((obs_id, action_map_w, 1))
                if np.any(action_map_s+1):
                    append((obs_id, action_map_s, 2))
                if np.any(action_map_s+1):
                    append((obs_id, action_map_e, 3))

    return obses, samples



episode_dir = '../UNet/full_episodes'
obses, samples = create_dataset_from_json(episode_dir)
print('obses:', len(obses), 'samples:', len(samples))


def make_input(obs):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    cities_opp = {}

    b = np.zeros((14, 32, 32), dtype=np.float32)
    b_global = np.zeros((15, 4, 4), dtype=np.float32)

    global_unit = 0
    global_rp = 0
    global_city = 0
    global_citytile = 0

    global_unit_opp = 0
    global_rp_opp = 0
    global_city_opp = 0
    global_citytile_opp = 0

    global_wood = 0
    global_coal = 0
    global_uranium = 0

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            team = int(strs[2])
            cooldown = float(strs[6])
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if team == obs['player']:
#################################### MAKE SURE THE ORDER OF X,Y IS CORRECT!!! ########################
                b[0, y, x] = 1  # b0 friend unit
                global_unit += 1
                b[1, y, x] = cooldown / 6  # b1 friend cooldown
                b[2, y, x] = (wood + coal + uranium) / 100  # b2 friend cargo
            else:
                b[3, y, x] = 1  # b3 oppo unit
                global_unit_opp += 1
                b[4, y, x] = cooldown / 6  # b4 oppo cooldown
                b[5, y, x] = (wood + coal + uranium) / 100  # b5 oppo cargo

        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            if team == obs['player']:
                global_citytile += 1
                b[6, y, x] = 1  # b6 friend city
                b[7, y, x] = cities[city_id]  # b7 friend city nights to survive
            else:
                global_citytile_opp += 1
                b[8, y, x] = 1  # b8 oppo city
                b[9, y, x] = cities_opp[city_id]  # b9 oppo city nights to survive
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 10, 'coal': 11, 'uranium': 12}[r_type], y, x] = amt / 800
            if r_type == 'wood':
                global_wood += 1
            elif r_type == "coal":
                global_coal += 1
            else:
                global_uranium += 1
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            if team == obs['player']:
                global_rp = min(rp, 200) / 200
            else:
                global_rp_opp = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            team = int(strs[1])
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            if team == obs['player']:
                global_city += 1
                cities[city_id] = min(fuel / lightupkeep, 20) / 20
            else:
                global_city_opp += 1
                cities_opp[city_id] = min(fuel / lightupkeep, 20) / 20
    # Map Size
    b[13, y_shift:32 - y_shift, x_shift:32 - x_shift] = 1
    # global features (normalized)
    b_global[0, :, :] = global_unit / width / height
    b_global[1, :, :] = global_rp
    b_global[2, :, :] = global_city / width / height
    b_global[3, :, :] = global_citytile / width / height
    b_global[4, :, :] = np.array(list(cities.values())).mean() if cities else 0
    b_global[5, :, :] = global_unit_opp / width / height
    b_global[6, :, :] = global_rp_opp
    b_global[7, :, :] = global_city_opp / width / height
    b_global[8, :, :] = global_citytile_opp / width / height
    b_global[9, :, :] = np.array(list(cities_opp.values())).mean() if cities_opp else 0
    b_global[10, :, :] = global_wood / width / height
    b_global[11, :, :] = global_coal / width / height
    b_global[12, :, :] = global_uranium / width / height
    b_global[13, :, :] = obs['step'] % 40 / 40  # Day/Night Cycle
    b_global[14, :, :] = obs['step'] / 360  # Turns

    return b, b_global


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, action_map, direction = self.samples[idx]
        obs = self.obses[obs_id]
        state_1, state_2 = make_input(obs)
        if direction == 0:
            return state_1, state_2, action_map
        else:
            # rotate the state & action map according to the direction(=0,1,2,3)
            state_1 = np.rot90(state_1,direction,(1,2)).copy()
            action_map = np.rot90(action_map,direction,(0,1)).copy()
            return state_1, state_2, action_map


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    global best_acc
    global global_epoch
    global_epoch += 1

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0
            epoch_num = 0
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states_1 = item[0].cuda().float()
                states_2 = item[1].cuda().float()
                actions = item[2].cuda().long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states_1, states_2)

                    loss = criterion(policy, actions)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(actions == policy.argmax(dim=1))

                    epoch_num += torch.sum(actions >= 0)
            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / epoch_num

            print(f'Epoch {global_epoch}/100 | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        if epoch_acc >= best_acc:
            print('save model...')
            traced = torch.jit.trace(model.cpu(), (torch.rand(1, 14, 32, 32), torch.rand(1, 15, 4, 4)))
            traced.save('model_full_trick_CE.pth')
            best_acc = epoch_acc


from unet_model import UNet

model = UNet(14, 3, 15)
train, val = train_test_split(samples, test_size=0.1, random_state=42)
batch_size = 256
train_loader = DataLoader(
    LuxDataset(obses, train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    LuxDataset(obses, val),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
dataloaders_dict = {"train": train_loader, "val": val_loader}

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
best_acc = 0.0
global_epoch = 0
for n in range(40):
    print("Learnint with lr :", optimizer.state_dict()['param_groups'][0]['lr'])
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=1)
    scheduler.step()