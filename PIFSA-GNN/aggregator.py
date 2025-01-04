import torch
import torch.nn.functional as F

class Aggregator(torch.nn.Module):

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):

        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors_swap_attention(neighbor_vectors, neighbor_relations, user_embeddings)
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
        else:
            output = torch.max(self_vectors, neighbors_agg).view((-1, self.dim))
            
        output = self.weights(output)

        return output

    def _mix_neighbor_vectors_swap_attention(self, neighbor_vectors, neighbor_relations, user_embeddings):

        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))
        user_embeddings_normalized = F.normalize(user_embeddings, dim=-1)
        neighbor_vectors_normalized = F.normalize(neighbor_vectors, dim=-1)
        neighbor_relations_normalized = F.normalize(neighbor_relations, dim=-1)
        relation_scores = (user_embeddings_normalized * neighbor_relations_normalized).sum(dim=-1)
        similarity_scores = (user_embeddings_normalized * neighbor_vectors_normalized).sum(dim=-1)
        user_relation_scores_normalized = torch.sigmoid(relation_scores)
        user_similarity_scores_normalized = torch.sigmoid(similarity_scores)
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)
        user_similarity_scores_normalized = user_similarity_scores_normalized.unsqueeze(dim=-1)
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors +
                                user_similarity_scores_normalized * neighbor_relations).sum(dim=2)

        return neighbors_aggregated

