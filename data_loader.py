
from ast import dump
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from utils import *
import scipy.sparse as sparse
import networkx as nx
import torch
import random
import pickle



n_item=0
n_user=0
n_relations = 0
n_interaction= 0
n_entity= 0

def get_inter_mat(file,args):
    # load cf data from [dataset].inter
    global n_user, n_item,n_interaction
    user_set = {}
    item_set = {}
    inter_dict={}
    id_set=set([])
    for line in open(file.replace('inter','link'),'r',encoding='utf-8').readlines()[1:]:
        array = line.replace('\n','').split('\t')
        item = array[0]
        id_set.add(item)
    
    
    inter_mat =[]
    u_i =[]
    i_i =[]
    values = []
    for line in open(file,'r',encoding='utf-8').readlines()[1:]:
        array = line.replace('\n','').split('\t')
        user = array[0]
        item = array[1]
        rating = array[2]
        if item in id_set:
            # remap user id
            if user in user_set.keys():
                user_id = user_set[user] 
            else:
                user_id = n_user
                user_set[user] = n_user
                n_user+=1
            # rem,ap item id
            if item in item_set.keys():
                item_id = item_set[item] 
            else:
                item_id = n_item
                item_set[item]  = n_item
                n_item+=1

            # get the catergories of rating interactions
            if args.dataset =='music':
                inter_type = get_interact_type_music(eval(rating))
            elif args.dataset =='Last-FM':
                inter_type = get_interact_type_Last_FM(eval(rating))
            else:
                inter_type = get_interact_type_movie(eval(rating))

            # build rating interaction matrix
            inter_mat.append([int(user_id),int(item_id),int(inter_type)])
            
            if int(inter_type)>n_interaction:
                n_interaction = int(inter_type)
            u_i.append(int(user_id))
            i_i.append(int(item_id))
            values.append(int(inter_type))
            inter_dict[(user_id,item_id)] = inter_type
            
    i = torch.LongTensor([u_i, i_i])
    v = torch.LongTensor(values)
    res = np.array(inter_mat)
    return res,user_set,item_set,inter_dict
    
def get_user_inter_set(inter_mat):
    user_inter_set = defaultdict(list)
    for u_id,i_id,type in inter_mat:
        user_inter_set[int(u_id)].append(int(i_id)) # {[item_id,item_id],...,[item_id,item_id]}
    return user_inter_set

def get_kg(kg_file, link_file,item_set):
    global n_entity, n_item, n_relations
    n_entity = n_item
    kg =[]
    triplet_set = {}
    # a map for item to entity
    i2e_link = {}
    for line in open(link_file,'r',encoding='utf-8').readlines()[1:]:
        array = line.replace('\n','').split('\t')
        item_id = array[0]
        entity_id = array[1]
        if item_id not in i2e_link.keys():
            i2e_link[item_id]=entity_id
    e2i_link = dict([val,key] for key,val in i2e_link.items())


    # a map for relation to relation id：
    relation2id ={}
    for line in tqdm(open(kg_file,'r',encoding='utf-8').readlines()[1:]):
        array = line.replace('\n','').split('\t')
        relation = array[1]
        if relation not in relation2id.keys():
            relation2id[relation]=len(list(relation2id.keys()))

    # process KG
    for line in tqdm(open(kg_file,'r',encoding='utf-8').readlines()[1:]):
        array = line.replace('\n','').split('\t')
        entity_1 = array[0]
        entity_2 = array[2]


        if entity_1 in e2i_link.keys():
            if e2i_link[entity_1] in item_set.keys():
                triplet_set[item_set[e2i_link[entity_1]]] = entity_1
                entity_1 = item_set[e2i_link[entity_1]] 
            else:
                item_set[e2i_link[entity_1]] = n_entity
                triplet_set[item_set[e2i_link[entity_1]]] = entity_1
                entity_1 = n_entity
                n_entity +=1         
        else:
            triplet_set[n_entity] =entity_1
            e2i_link[entity_1] = n_entity
            item_set[e2i_link[entity_1]]=n_entity
            entity_1 = n_entity
            n_entity +=1

        if entity_2 in e2i_link.keys():
            if e2i_link[entity_2] in item_set.keys():
                triplet_set[item_set[e2i_link[entity_2]]] =entity_2
                entity_2 = item_set[e2i_link[entity_2]]
            else:
                item_set[e2i_link[entity_2]]=n_entity  
                triplet_set[item_set[e2i_link[entity_2]]] =entity_2
                entity_2 = n_entity
                n_entity +=1  
        else:
            triplet_set[n_entity] =entity_2
            e2i_link[entity_2] = n_entity
            item_set[e2i_link[entity_2]]=n_entity 
            entity_2 = n_entity
            n_entity +=1



        r_id = relation2id[array[1]]
        if int(r_id)>n_relations:
            n_relations = int(r_id)
        kg.append([int(entity_1),int(r_id),int(entity_2)])
    res = np.array(kg)
    return res, triplet_set



def build_graph(kg):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(set)
    all_h =[]
    all_t =[]
    all_r =[]

    
    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(kg, ascii=True):
        all_h.append(h_id)
        all_t.append(t_id)
        all_r.append(r_id)
        ckg_graph.add_edge(h_id,t_id,key=r_id)
        rd[(h_id,r_id)].add(t_id)
    return ckg_graph, rd


def build_sparse_relation_graph(relation_dict):

    def _bi_norm_lap(adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()        
        d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()
    

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum,-1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sparse.diags(d_inv)
        
        si_lap = d_mat_inv.dot(adj)
        return si_lap.tocoo()


    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:,1] = cf[:,1] + n_user
            vals = [1.] * len(cf)
            adj = sparse.coo_matrix((vals, (cf[:,0],cf[:,1])), shape = (n_entity+n_user, n_entity+n_user))
        else:
            vals = [1.] * len(np_mat)
            adj = sparse.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_entity+n_user, n_entity+n_user))
        
        adj_mat_list.append(adj)
    
    norm_mat_list =  [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list =  [_si_norm_lap(mat) for mat in adj_mat_list]

    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_user,n_user:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_user,n_user:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list



def get_sparse_inter_adj_mat(train_cf,inter_dict):
    col = np.zeros(len(train_cf))
    row = np.zeros(len(train_cf))
    value = np.zeros(len(train_cf))
    j = 0
    for i in tqdm(train_cf):
        row[j] = i[0]
        col[j] = i[1]
        value[j] = int(inter_dict[(i[0],i[1])])
        j+=1
    row= np.array(row,dtype=int)
    col= np.array(col,dtype=int)
    value= np.array(value,dtype=int)
    return sparse.coo_matrix((value,(row,col)))


# split dataset
def split_dataset(user_inter_set,train_rate=0.8,shuffle=True):
    train_set = defaultdict(list)
    test_set = defaultdict(list)
    for user in user_inter_set.items():
        n_total = len(user[1])
        offset =int(n_total * train_rate)
        if n_total == 0 or offset < 1:
            if n_total >= 1:
                offset = n_total
        if shuffle:
            random.shuffle(user[1])
        train_set[user[0]] = sorted(user[1][:offset])
        test_set[user[0]] = sorted(user[1][offset:n_total])

    user_dict={
        'train_user_set':train_set,
        'test_user_set':test_set
    }
    train_cf =[]
    test_cf = []
    for t in train_set.items():
        for i in t[1]:
            train_cf.append([t[0],i])
    for t in test_set.items():
        for i in t[1]:
            test_cf.append([t[0],i])

    return user_dict,train_cf,test_cf


        

def load_data(args):
    
    file  = './dataset/'+args.dataset+'/'+args.dataset+'.inter'
    kg_file = './dataset/'+args.dataset+'/'+args.dataset+'.kg'
    link_file = './dataset/'+args.dataset+'/'+args.dataset+'.link'


    print("reading interaction data ...")
    inter_mat,user_set,item_set,inter_dict = get_inter_mat(file,args)
    user_inter_set = get_user_inter_set(inter_mat)

    with open('./dataset/'+args.dataset+'/user_map.pkl','wb') as f:
        pickle.dump(user_set,f)

    with open('./dataset/'+args.dataset+'/item_map.pkl','wb') as f:
        pickle.dump(item_set,f)  

    print("reading kg data ...")
    kg, entity_set = get_kg(kg_file,link_file,item_set)


    with open('./dataset/'+args.dataset+'/entity_map.pkl','wb') as f:
        pickle.dump(entity_set,f)  

    print("split data ...")
    user_dict,train_cf,test_cf = split_dataset(user_inter_set)

    print("build graph ...")
    graph, relation_dict = build_graph(kg)

    print("get sparse inter_mat ...")
    inter_adj_mat = get_sparse_inter_adj_mat(train_cf,inter_dict)

    n_params = {}
    n_params['n_users'] = n_user
    n_params['n_items'] = n_item
    n_params['n_entity'] = n_entity
    n_params['n_relations'] = n_relations+1
    n_params['n_interactions'] = n_interaction+1
    


    return train_cf, test_cf,  user_dict, n_params, graph,relation_dict, inter_adj_mat, inter_dict
 

