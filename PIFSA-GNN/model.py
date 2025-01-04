import torch
import random
from tqdm import tqdm
from aggregator import Aggregator

class PIFSA_GNN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device, UAndI):
        super(PIFSA_GNN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.UAndI = UAndI
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)
        self._gen_adj()
        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        self.conv = torch.nn.Conv2d(self.n_neighbor,1,3,1,1)

    def _gen_adj(self):

        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        
        for e in tqdm(self.kg, total=len(self.kg), desc='Generate KG adjacency matrix:'):
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])


    def forward(self, u, v):
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        user_and_item_adj = torch.empty(self.batch_size, self.n_neighbor, dtype=torch.long).to(self.device)
        ki = 0
        for user in u:
            if user.item() in self.UAndI:
                user_and_item_adj[ki] = torch.LongTensor(self.UAndI[user.item()])
            else:
                user_and_item_adj[ki] = torch.LongTensor([0] * self.n_neighbor)
            ki = ki+1

        UAndIEmmbbing = self.ent(user_and_item_adj)

        u = u.view((-1, 1))
        v = v.view((-1, 1))

        user_embeddings = self.usr(u)
        user_embeddings = user_embeddings*UAndIEmmbbing
        user_embeddings = user_embeddings.unsqueeze(dim=3)
        user_embeddings = self.conv(user_embeddings).squeeze()
        entities, relations = self._get_neighbors(v)
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        scores = (user_embeddings * item_embeddings).sum(dim = 1)
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):

        entities = [v]
        relations = []
        neighbor_entities = torch.LongTensor(self.adj_ent[entities[0].cpu()]).view((self.batch_size, -1)).to(self.device)
        neighbor_relations = torch.LongTensor(self.adj_rel[entities[0].cpu()]).view((self.batch_size, -1)).to(self.device)
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations):

        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]
        vector = self.aggregator(
            self_vectors=entity_vectors[0],
            neighbor_vectors=entity_vectors[1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
            neighbor_relations=relation_vectors[0].view((self.batch_size, -1, self.n_neighbor, self.dim)),
            user_embeddings=user_embeddings)
        return vector


