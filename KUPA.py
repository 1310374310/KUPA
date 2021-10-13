from functools import reduce
import numpy as np
import torch
from torch._C import device 
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
from torch_scatter import scatter_mean,scatter_add,scatter_softmax

class Aggregator(nn.Module):
    def __init__(self,n_users,n_interactions):
        super(Aggregator,self).__init__()
        self.n_users=n_users
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.linear = nn.Linear(64, 1)
        self.project = nn.Linear(64,64)
        self.tanh = nn.Tanh()


    def forward(self,entity_emb,user_emb,interaction_emb, relation_emb, edge_index, 
                edge_type,interact_mat,agg_method='attention'):
                                
        n_entities = entity_emb.shape[0]
        n_users = self.n_users

        # Knowledge-awar propogation
        head,tail = edge_index
        neigh_relation_emb = entity_emb[tail]
        w1 = (entity_emb[tail]*relation_emb[edge_type-1]).sum(1).squeeze()/8
        w1 = torch.unsqueeze(scatter_softmax(w1, index=head),1)
        entity_agg = scatter_add(src=neigh_relation_emb*w1, index=head, dim_size=n_entities, dim=0)

        
        if agg_method == 'GNN':
            user_index =  interact_mat._indices()[0]
            item_index =  interact_mat._indices()[1]
            interact_type =  interact_mat._values()
            interact_entity_emb = entity_emb[item_index]
            interact_score = self.linear(interact_entity_emb)
            user_agg = scatter_mean(src=interact_score, index=user_index, dim_size=n_users,dim=0)

            att_w = (interaction_emb[interact_type]*user_emb[user_index]).sum(1).squeeze()
            att_w =  torch.unsqueeze(scatter_softmax(att_w,index=user_index),1)
        

        #  Prefernce-aware propogation
        elif agg_method =='attention':
            user_index =  interact_mat._indices()[0]
            item_index =  interact_mat._indices()[1]
            interact_type =  interact_mat._values()
            interact_entity_emb = entity_emb[item_index]
            #[n_interactions,1]
            att_w = (interaction_emb[interact_type]*user_emb[user_index]*interact_entity_emb).sum(1).squeeze()
            att_w =  torch.unsqueeze(scatter_softmax(att_w,index=user_index),1)
            interact_score = att_w*interact_entity_emb
            user_agg = scatter_add(src=interact_score, index=user_index, dim_size=n_users,dim=0)

        return entity_agg, user_agg, (att_w, interact_mat, w1, edge_index)



class KGConv(nn.Module):
    def __init__(self,n_hops,n_users,node_dropout_rate,interact_mat,agg,mess_dropout_rate=0.2):
        super(KGConv,self).__init__()
        self.agg = agg
        self.conv = nn.ModuleList()
        self.dropout = nn.Dropout2d(0.1)
        self.n_interactions = interact_mat[0].shape[0]
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        for _ in range(n_hops):
            self.conv.append(Aggregator(n_users=self.n_users,n_interactions=self.n_interactions))
        self.dropout = nn.Dropout(p=mess_dropout_rate)


    def forward(self, user_emb, entity_emb, interact_emb,edge_index,edge_type,interact_mat,relation_emb, mess_dropout=True,node_dropout=True):


        entity_res_emb = entity_emb
        user_res_emb = user_emb
        temp = 0
        for i in range(len(self.conv)):
            entity_emb,user_emb, att_w = self.conv[i](entity_emb=entity_emb,
                                              user_emb=user_emb,
                                              interaction_emb=interact_emb,
                                              relation_emb=relation_emb,
                                              edge_index = edge_index,
                                              edge_type= edge_type,
                                              interact_mat=interact_mat,
                                              agg_method = self.agg)
            if i==3:
                temp = att_w
            if mess_dropout:
                entity_emb =  self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, temp
     
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]


    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask].long()
        out = torch.sparse.LongTensor(i, v, x.shape).to(x.device)
        #print(out._values())
        return out

class KUPA(nn.Module):

    def __init__(self, args,n_params, adj_mat, kg):
        super(KUPA,self).__init__()

        # init model
        self.n_users = n_params['n_users']
        self.n_entity = n_params['n_entity']
        self.n_items = n_params['n_items']
        self.n_relation = n_params['n_relations']
        self.n_interactions = n_params['n_interactions']
        self.dim = args.dim
        self.decay =args.l2
        self.lu=args.lu
        self.ls= args.ls
        self.node_dropout_rate = args.node_dropout_rate
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.edge_index, self.edge_type = self._get_edges(kg)
        self.interact_mat = self._get_interact(adj_mat)
        

        self._init_weight()
        self.entity_emb = nn.Parameter(self.entity_emb)
        self.relation_emb =  nn.Parameter(self.relation_emb)
        self.user_emb =  nn.Parameter(self.user_emb)
        self.interact_emb =  nn.Parameter(self.interact_emb)

        # propogation Layer
        self.kgcn = self._init_KGCN(args)
        self.tanh = nn.Tanh()


    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.entity_emb = initializer(torch.empty(self.n_entity, self.dim))
        self.user_emb = initializer(torch.empty(self.n_users, self.dim))
        self.relation_emb = initializer(torch.empty(self.n_relation, self.dim))
        self.interact_emb = initializer(torch.empty(self.n_interactions, self.dim))

        
    def _init_KGCN(self,args):
        return KGConv(n_hops=args.n_hops,
                      n_users=self.n_users,
                      interact_mat=self.interact_mat,
                      node_dropout_rate=args.node_dropout_rate,
                      agg=args.agg)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        etype = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), etype.long().to(self.device)


    def _get_interact(self,adj_mat):
        i = torch.LongTensor([adj_mat.row,adj_mat.col]).to(self.device)
        v = torch.LongTensor(adj_mat.data).to(self.device)
        return torch.sparse.FloatTensor(i, v, adj_mat.shape)


    def forward(self,batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        entity_kgcn_emb, user_kgcn_emb, _ = self.kgcn(user_emb = self.user_emb,
                                                  entity_emb = self.entity_emb,
                                                  relation_emb = self.relation_emb,
                                                  interact_emb = self.interact_emb,
                                                  edge_index = self.edge_index,
                                                  edge_type =  self.edge_type,
                                                  interact_mat = self.interact_mat)


        u_e = self.tanh(user_kgcn_emb[user])
        pos_e = self.tanh(entity_kgcn_emb[pos_item])
        neg_e = self.tanh(entity_kgcn_emb[neg_item])
        interact_e = self.tanh(self.interact_emb)


        BPR_loss = self._BPR_loss(u_e, pos_e, neg_e)
        user_loss = self._user_loss(u_e, interact_e, pos_e, neg_e)*self.lu
        se_loss = self._se_Loss(u_e,interact_e)*self.ls
        #rel_loss = self._rel_loss(self.relation_emb)


        loss = BPR_loss + se_loss + user_loss
        #loss = BPR_loss+user_loss
        return loss, se_loss, user_loss


    def generate(self):
        entity_kgcn_emb,user_kgcn_emb, _ = self.kgcn(user_emb = self.user_emb,
                                                  entity_emb = self.entity_emb,
                                                  relation_emb = self.relation_emb,
                                                  interact_emb = self.interact_emb,
                                                  edge_index = self.edge_index,
                                                  edge_type =  self.edge_type,
                                                  interact_mat = self.interact_mat)
        return entity_kgcn_emb, user_kgcn_emb





    def _user_loss(self,u_emb,interact_emb,p_emb,n_emb):
        user_loss= 0
        for i in range(1,interact_emb.shape[0]):
            #for j in range(i + 1, self.n_interactions):
                neg_scores = torch.sum(torch.mul(u_emb, interact_emb[i-1]), axis=1)
                pos_scores = torch.sum(torch.mul(u_emb, interact_emb[i]), axis=1)
                user_loss += -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        return user_loss/(i)



    def _BPR_loss(self,u_emb,p_emb,n_emb):
        batch_size = u_emb.shape[0]
        pos_scores = torch.sum(torch.mul(u_emb, p_emb), axis=1)
        neg_scores = torch.sum(torch.mul(u_emb, n_emb), axis=1)
        
        # BPR
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(u_emb) ** 2
                       + torch.norm(p_emb) ** 2
                       + torch.norm(n_emb) ** 2) 
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss
      
                
    def _se_Loss(self,u_emb,interact_emb):

        normal_interact_emb = interact_emb / interact_emb.norm(dim=1,keepdim=True)
        normal_interact_emb_T = normal_interact_emb.t()
        
        pos_scores = 0

        for i in range(1,normal_interact_emb.shape[0]):
            for j in range(i+1,normal_interact_emb.shape[0]):
                pos_scores+= torch.mul(normal_interact_emb[i],normal_interact_emb[j])

        pos_scores = torch.sum(normal_interact_emb_T * normal_interact_emb_T, dim=1)
        pos_scores = pos_scores/(i*j/2)
        ttl_scores = torch.sum(torch.mm(normal_interact_emb_T, normal_interact_emb), dim=1)

        pos_scores = torch.exp(pos_scores / 0.2)
        ttl_scores = torch.exp(ttl_scores / 0.2)
        mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
        return mi_score

        
    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())