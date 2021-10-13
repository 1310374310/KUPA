from data_loader import load_data
import random
import torch
import numpy as np
from time import time
import time
from KUPA import KUPA
from prettytable import PrettyTable
from evaluate import test
from utils import early_stopping
from parser import parse_args
import logging

time_stamp = int(time.time())
time_array = time.localtime(time_stamp)
str_date = time.strftime("%Y-%m-%d %H:%M:%S", time_array)

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename= './log/'+str_date+'.log',
                    filemode='w')

def negative_tail_sampling(kg,rd,n_entity):
    graph_tensor = torch.tensor(list(kg.edges))  # [-1, 3]
    index = graph_tensor[:, :-1]  # [-1, 2]
    type = graph_tensor[:, -1]  # [-1, 1]
    neg_tails=[]
    hs, ts = index.t().long()
    for i,h in enumerate(hs):
        r = type[i]
        while True:
            neg_tail = np.random.randint(low=0, high=n_entity, size=1)[0]
            if neg_tail not in rd[(int(h),r)]:
                break
        neg_tails.append(neg_tail)
    return neg_tails

def train(args):
    # enure that the model results are reproducible
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    # load data
    train_cf, test_cf, user_dict, n_params, kg, relation_dict, inter_adj_mat,inter_dict = load_data(args)
    
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_interactions = n_params['n_interactions']
    n_relation = n_params['n_relations']
    n_entity = n_params['n_entity']
    logging.info("#"*50)
    for k in list(args.__dict__):
        logging.info('%s: %s' %(k, args.__dict__[k]))
    logging.info("#"*50)


    logging.info("n_users: %d" %n_users)
    logging.info("n_items: "+str(n_items))
    logging.info("n_interactions: "+str(n_interactions))
    logging.info("n_relation: "+str(n_relation))
    logging.info("n_entity: "+str(n_entity))
    
    logging.info("dataset: "+str(args.dataset))
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    #test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    
    
    model = KUPA(args,n_params=n_params,
                    adj_mat= inter_adj_mat,
                    kg=kg)

    logging.info(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False


    print("start training ...")
    for epoch in range(args.n_epoch):
        neg_tails=0
        index = np.arange(len(train_cf))
        #logging.info("shuffle data ...")
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        model.train()
        loss, s = 0, 0
        inter_loss = 0
        kg_loss = 0
        train_s_t = time.time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'],
                                  neg_tails,
                                  n_items,
                                  device)
            batch_loss,batch_inter,batch_kg = model(batch)

            batch_loss = batch_loss
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            inter_loss +=batch_inter
            kg_loss +=batch_kg

            s += args.batch_size

        train_e_t = time.time()
        torch.cuda.empty_cache()
        


        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            model.eval()
            test_s_t = time.time()
            ret = test(model, user_dict, n_params)
            test_e_t = time.time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio","auc"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'], ret['auc']]
            )
            print(train_res)
            logging.info(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][1], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][1] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')


        else:
            print('using time %.4f, training loss at epoch %d: %.4f, inter_loss %.4f, kg_loss %.4f' % (train_e_t - train_s_t, epoch, loss.item(), inter_loss.item(), kg_loss))
            logging.info('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

        

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    logging.info('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))



            

def get_feed_dict(train_entity_pairs,start,end,train_user_set,neg_tails,n_items,device):

    def negative_sampling(user_item,train_user_set):
        neg_items=[]
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items
    

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)

    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


    
if __name__=='__main__':
    args = parse_args()
    train(args)