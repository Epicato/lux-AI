import os
import numpy as np
import torch
from lux.game import Game
import torch.nn.functional as F

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'

model = torch.jit.load(f'{path}/model_full_trick_CE.pth')
model.eval()


def make_input(obs, game_state):
    width, height = game_state.map.width, game_state.map.height
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

    # global_wood_amount = 0
    # global_coal_amount = 0
    # global_uranium_amount = 0

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
            if team == obs.player:
#################################### MAKE SURE THE ORDER OF X,Y IS CORRECT!!! ########################
################################# This is important if you want to rotate the map.####################
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
            if team == obs.player:
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
            if team == obs.player:
                global_rp = min(rp, 200) / 200
            else:
                global_rp_opp = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            team = int(strs[1])
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            if team == obs.player:
                global_city += 1
                # let the agent care about whether the city can survive for 20 nights
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


game_state = None


def get_game_state(observation):
    global game_state

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    return game_state


def get_shift(game_state):
    width, height = game_state.map.width, game_state.map.height
    shift = (32 - width) // 2
    return shift


def in_city(pos):
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


### 0:n 1:w 2:s 3:e 4:bc 5:c
unit_actions = [('move', 'n'), ('move', 'w'), ('move', 's'), ('move', 'e'), ('build_city',), ('move', 'c')]


def get_action(policy, unit, dest, shift):
    p = policy[:, unit.pos.y + shift, unit.pos.x + shift]

    for label in np.argsort(p)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos
        if pos not in dest or in_city(pos):
            return call_func(unit, *act), pos

    return unit.move('c'), unit.pos


def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)
    shift = get_shift(game_state)
    player = game_state.players[observation.player]
    actions = []

    # City Actions
    unit_count = len(player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                if unit_count < player.city_tile_count:
                    actions.append(city_tile.build_worker())
                    unit_count += 1
                elif not player.researched_uranium():
                    actions.append(city_tile.research())
                    player.research_points += 1

    # Worker Actions
    state_1, state_2 = make_input(observation, game_state)
    dest = []
    with torch.no_grad():
        # 0:north 1:build_city 2:center(transfer)
        # predict one time for each direction, so in one turn we have to predict 4 times
        p_n = model(torch.from_numpy(state_1).unsqueeze(0), torch.from_numpy(state_2).unsqueeze(0)).squeeze(0)

        p_n = p_n.numpy()

        # rotate the input state for 3 times and send it to our model
        obs_e = torch.rot90(torch.from_numpy(state_1), 1, (1, 2)).unsqueeze(0)
        obs_s = torch.rot90(torch.from_numpy(state_1), 2, (1, 2)).unsqueeze(0)
        obs_w = torch.rot90(torch.from_numpy(state_1), 3, (1, 2)).unsqueeze(0)

        p_e = model(obs_e, torch.from_numpy(state_2).unsqueeze(0)).squeeze(0)

        p_e = p_e.numpy()
        p_e = np.rot90(p_e, -1, (1, 2))     # action map for e

        p_s = model(obs_s, torch.from_numpy(state_2).unsqueeze(0)).squeeze(0)

        p_s = p_s.numpy()
        p_s = np.rot90(p_s, -2, (1, 2))     # action map for s

        p_w = model(obs_w, torch.from_numpy(state_2).unsqueeze(0)).squeeze(0)

        p_w = p_w.numpy()
        p_w = np.rot90(p_w, 1, (1, 2))      # action map for w

        p_bc = (p_n[1]+p_w[1]+p_s[1]+p_e[1])/4  # mean of action: build_city
        p_c = (p_n[2]+p_w[2]+p_s[2]+p_e[2])/4   # mean of action: center(or transfer resources)

        # 0:n 1:w 2:s 3:e 4:bc 5:c
        policy = np.stack((p_n[0], p_w[0], p_s[0], p_e[0], p_bc, p_c))


    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            action, pos = get_action(policy, unit, dest, shift)
            actions.append(action)
            dest.append(pos)

    return actions
