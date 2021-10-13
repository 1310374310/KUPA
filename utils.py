from collections import defaultdict
import os
import re
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)




def get_interact_type_music(rating):
    if rating>10000:
        return 4
    elif rating>5000:
        return 3
    elif rating>1000:
        return 2
    elif rating>100:
        return 1
    else:
        return 0
def get_interact_type_Last_FM(rating):
    if rating>700:
        return 4
    elif rating>500:
        return 3
    elif rating>300:
        return 2
    elif rating>100:
        return 1
    else:
        return 0

def get_interact_type_book(rating):
    return int(rating/2)

def get_interact_type_movie(rating):
    return int(rating)

def reverse_map(data):
    return {value:key for key,value in data.items()}

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def trans_graph2dict(kg):
    res = defaultdict(list)
    for edge in kg.edges:
        h = edge[0]
        t = edge[1]
        r = edge[2]
        res[h].append((t,r))
    return res


def trans_pairs2dict(pairs,item_set):
    user2item_dict  = defaultdict(list)
    item2user_dict = defaultdict(list)
    #item_set = set()
    for p in pairs:
        if p[0] not in user2item_dict:
            user2item_dict[p[0]]=[]
        user2item_dict[p[0]].append(p[1])

        if p[1] not in item2user_dict:
            item2user_dict[p[1]]=[]
        item2user_dict[p[1]].append(p[0])

        #item_set.add(p[1])


    item2item_dict =  defaultdict(list)
    for item in item2user_dict.keys():
        item_neighbor_item = []
        for user in item2user_dict[item]:
            item_neighbor_item = item_neighbor_item + user2item_dict[user]
        item2item_dict[item] = list(set(item_neighbor_item))
    
    for item in item_set:
        if  item not in item2item_dict:
            item2item_dict[item] = [item]

    return user2item_dict, item2item_dict



def get_triplet_set(args, kg, init_triplet_set, set_size, is_user):
    triple_sets =  defaultdict(list)
    for obj in tqdm(init_triplet_set.keys()):
        if is_user and args.n_hops == 0:
            n_layer = 1
        else:
            n_layer = args.n_hops

        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_triplet_set[obj]
            else:
                entities = triple_sets[obj][-1][2]
            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])

            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets
