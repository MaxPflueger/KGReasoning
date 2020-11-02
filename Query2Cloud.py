import logging
import collections
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from geomloss import SamplesLoss
from tqdm import tqdm

class Score_Func(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_func = SamplesLoss(loss='sinkhorn', blur=.05, p=2)

    def forward(self, embeds1, embeds2, mode='batch'):
        if embeds1.shape == embeds2.shape:
            return self.score_func(embeds1, embeds2)
        elif mode == 'batch':
            # If batchsize is large in comparison to number of negative samples - usually the case
            distances = torch.zeros(embeds2.shape[0], embeds2.shape[1])
            embeds2 = embeds2.transpose(0,1)
            for i in range(embeds2.shape[0]):
                    distances[:,i] = self.score_func(embeds1, embeds2[i])
            return distances
        else:
            # If batchsize is small in comparison to number of negative samples
            distances = torch.zeros(embeds2.shape[0], embeds2.shape[1])
            for i in range(embeds2.shape[0]):
                    h = embeds2[i]
                    g = embeds1[i].repeat(embeds2.shape[1], 1, 1)
                    res = self.score_func(g, h)
                    distances[i,:] = res
            return distances


class CloudEncoder(nn.Module):
    def __init__(self, nentity, embedding_dim, n_vec):
        super().__init__()
        self.entity_embeddings = nn.Embedding(nentity, embedding_dim * n_vec)
        self.n_vec = n_vec
        self.embedding_dim = embedding_dim


    def forward(self, indices):
        embeds = self.entity_embeddings(indices)
        embeds = embeds.reshape(-1, self.n_vec, self.embedding_dim)
        norm = embeds.norm(p=2, dim=2, keepdim=True)
        return embeds.div(norm.expand_as(embeds))


class CloudProjection(nn.Module):
    def __init__(self, nrels, embedding_dim):
        super(CloudProjection, self).__init__()
        self.mats = nn.Parameter(torch.FloatTensor(nrels, embedding_dim, embedding_dim))
        init.xavier_uniform_(self.mats)

    def forward(self, embeds, rels):
        rel_mats = torch.index_select(self.mats, dim=0, index=rels)
        embeds = torch.bmm(embeds, rel_mats)
        norm = embeds.norm(p=2, dim=2, keepdim=True)
        embeds = embeds.div(norm.expand_as(embeds))
        return embeds

class CloudIntersection(nn.Module):
    def __init__(self, embedding_dim, n_vec):
        super(CloudIntersection, self).__init__()
        self.n_vec = n_vec
        self.intra_cloud_agg = torch.max
        self.inter_cloud_agg = torch.mean
        self.pre_mats = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        init.xavier_uniform_(self.pre_mats)
        self.post_mats = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        init.xavier_uniform_(self.post_mats)
        self.pre_mats2 = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        init.xavier_uniform_(self.pre_mats2)
        self.post_mats2 = nn.Parameter(torch.FloatTensor(n_vec, embedding_dim, embedding_dim))
        init.xavier_uniform_(self.post_mats2)

    def forward(self, embeds1, embeds2, embeds3=None):
        embeds1 = F.leaky_relu(torch.bmm(embeds1, self.pre_mats.repeat(embeds1.shape[0],1,1)))
        embeds2 = F.leaky_relu(torch.bmm(embeds2, self.pre_mats.repeat(embeds2.shape[0],1,1)))
        agg_embeds1 = self.intra_cloud_agg(embeds1, dim=1)[0]
        agg_embeds2 = self.intra_cloud_agg(embeds2, dim=1)[0]
        agg_embeds1 = F.leaky_relu(torch.mm(agg_embeds1, self.post_mats))
        agg_embeds2 = F.leaky_relu(torch.mm(agg_embeds2, self.post_mats))
        if embeds3 != None:
            embeds3 = F.leaky_relu(torch.bmm(embeds3, self.pre_mats.repeat(embeds3.shape[0],1,1)))
            agg_embeds3 = self.intra_cloud_agg(embeds3, dim=1)[0]
            agg_embeds3 = F.leaky_relu(torch.mm(agg_embeds3, self.post_mats))
            intersect = self.inter_cloud_agg(torch.stack([agg_embeds1, agg_embeds2 , agg_embeds3]), dim=0)
        else:
            intersect = self.inter_cloud_agg(torch.stack([agg_embeds1,agg_embeds2]),dim=0)
        intersect = F.leaky_relu(torch.mm(intersect, self.pre_mats2))
        intersect = intersect.repeat(self.n_vec, 1, 1)
        intersect = torch.bmm(intersect, self.post_mats2)
        # Is this correct? Go through toy example
        intersect = intersect.transpose(1, 0)
        intersect = intersect.div(intersect.norm(p=2, dim=2, keepdim=True).expand_as(intersect))
        return intersect


class Query2Cloud(nn.Module):

    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None):
        super(Query2Cloud, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.n_vec = 8

        self.entity_embedding = CloudEncoder(self.nentity, self.entity_dim, self.n_vec)  # center for entities
        self.projection_operator = CloudProjection(self.nrelation, self.relation_dim)
        self.intersection_operator = CloudIntersection(self.entity_dim, self.n_vec)
        self.score_func = Score_Func()

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def calc_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                # embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                embedding = self.entity_embedding(queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.calc_query(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    # r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    # embedding += r_embedding
                    embedding = self.projection_operator(embedding, queries[:, idx])
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.calc_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            if len(embedding_list) < 3:
                embedding = self.intersection_operator(embedding_list[0], embedding_list[1])
            else:
                embedding = self.intersection_operator(embedding_list[0], embedding_list[1], embedding_list[2])
        return embedding, idx


    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.calc_query(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                 query_structure),
                                                      self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.calc_query(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            # all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 8, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                # positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_embedding = self.entity_embedding(positive_sample_regular)
                positive_logit = self.score_func(all_center_embeddings, positive_embedding)
            else:
                # positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
                positive_logit = torch.Tensor([]).to(self.entity_embedding.entity_embeddings.weight.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                # positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_embedding = self.entity_embedding(positive_sample_union)
                positive_union_logit = self.score_func(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                # positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.entity_embeddings.weight.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                # This is a work around because the geomloss library
                # negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                h = self.entity_embedding(negative_sample_regular.view(-1))
                negative_embedding = h.view(batch_size, negative_size, self.n_vec, -1)
                negative_logit = self.score_func(all_center_embeddings, negative_embedding)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.entity_embeddings.weight.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                # negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.entity_embedding(negative_sample_union.view(-1)).view(batch_size, negative_size, self.n_vec, -1)
                all_union_center_embeddings = all_union_center_embeddings.transpose(0,1)
                negative_union_logit = self.score_func(all_union_center_embeddings[0], negative_embedding).unsqueeze(0)
                for i in range(all_union_center_embeddings.shape[0]-1):
                    logit = self.score_func(all_union_center_embeddings[i+1], negative_embedding).unsqueeze(0)
                    negative_union_logit = torch.cat([negative_union_logit, logit], dim=0)
                negative_union_logit = torch.max(negative_union_logit, dim=0)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.entity_embeddings.weight.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def loss_log_sigmoid(positive_logit, negative_logit, subsampling_weight):
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
        return (positive_sample_loss + negative_sample_loss) / 2, positive_sample_loss, negative_sample_loss

    @staticmethod
    def margin_loss(positive_logit, negative_logit, subsampling_weight, margin = 1):
        negative_score = negative_logit.mean(dim=1)
        positive_score = positive_logit
        loss = margin + subsampling_weight * positive_score - (subsampling_weight * negative_score)
        return loss.mean(), positive_score.mean(), negative_score.mean()

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        print(step)
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample,
                                                                      subsampling_weight, batch_queries_dict,
                                                                      batch_idxs_dict)

        loss, positive_sample_loss, negative_sample_loss = model.margin_loss(positive_logit=positive_logit, negative_logit=negative_logit, subsampling_weight=subsampling_weight)
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(
                        argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics




