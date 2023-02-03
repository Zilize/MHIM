# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

r"""
PCR
====
References:
    Chen, Qibin, et al. `"Towards Knowledge-Based Recommender Dialog System."`_ in EMNLP 2019.

.. _`"Towards Knowledge-Based Recommender Dialog System."`:
   https://www.aclweb.org/anthology/D19-1189/

"""

import json
import os.path
import random

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv, HypergraphConv

from crslab.config import DATASET_PATH
from crslab.model.base import BaseModel
from crslab.model.crs.mhim.attention import MHItemAttention
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.model.crs.mhim.decoder import TransformerDecoderKG


class MHIMModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_entity: A integer indicating the number of entities.
        n_relation: A integer indicating the number of relation in KG.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        user_emb_dim: A integer indicating the dimension of user embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.
        user_proj_dim: A integer indicating dim to project for user embedding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.device = device
        self.gpu = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        assert self.dataset in ['HReDial', 'HTGReDial']
        # vocab
        self.pad_token_idx = vocab['tok2ind']['__pad__']
        self.start_token_idx = vocab['tok2ind']['__start__']
        self.end_token_idx = vocab['tok2ind']['__end__']
        self.vocab_size = vocab['vocab_size']
        self.token_emb_dim = opt.get('token_emb_dim', 300)
        self.pretrain_embedding = side_data.get('embedding', None)
        # kg
        self.n_entity = vocab['n_entity']
        self.entity_kg = side_data['entity_kg']
        self.n_relation = self.entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(self.entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get('num_bases', 8)
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim
        # transformer
        self.n_heads = opt.get('n_heads', 2)
        self.n_layers = opt.get('n_layers', 2)
        self.ffn_size = opt.get('ffn_size', 300)
        self.dropout = opt.get('dropout', 0.1)
        self.attention_dropout = opt.get('attention_dropout', 0.0)
        self.relu_dropout = opt.get('relu_dropout', 0.1)
        self.embeddings_scale = opt.get('embedding_scale', True)
        self.learn_positional_embeddings = opt.get('learn_positional_embeddings', False)
        self.reduction = opt.get('reduction', False)
        self.n_positions = opt.get('n_positions', 1024)
        self.longest_label = opt.get('longest_label', 30)
        self.user_proj_dim = opt.get('user_proj_dim', 512)
        # pooling
        self.pooling = opt.get('pooling', None)
        assert self.pooling == 'Attn' or self.pooling == 'Mean'
        # MHA
        self.mha_n_heads = opt.get('mha_n_heads', 4)
        self.extension_strategy = opt.get('extension_strategy', None)
        self.pretrain = opt.get('pretrain', False)
        self.pretrain_data = None
        self.pretrain_epoch = opt.get('pretrain_epoch', 9999)

        super(MHIMModel, self).__init__(opt, device)

    def build_model(self, *args, **kwargs):
        if self.pretrain:
            pretrain_file = os.path.join('pretrain', self.dataset, str(self.pretrain_epoch) + '-epoch.pth')
            self.pretrain_data = torch.load(pretrain_file, map_location=torch.device('cuda:' + str(self.gpu[0])))
            logger.info(f"[Load Pretrain Weights from {pretrain_file}]")
        self._build_copy_mask()
        self._build_adjacent_matrix()
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _build_copy_mask(self):
        token_filename = os.path.join(DATASET_PATH, "hredial", "nltk", "token2id.json")
        token_file = open(token_filename, 'r')
        token2id = json.load(token_file)
        id2token = {token2id[token]: token for token in token2id}
        self.copy_mask = list()
        for i in range(len(id2token)):
            token = id2token[i]
            if token[0] == '@':
                self.copy_mask.append(True)
            else:
                self.copy_mask.append(False)
        self.copy_mask = torch.as_tensor(self.copy_mask).to(self.device)

    def _build_adjacent_matrix(self):
        graph = dict()
        for head, tail, relation in tqdm(self.entity_kg['edge']):
            graph[head] = graph.get(head, []) + [tail]
        adj = dict()
        for entity in tqdm(range(self.n_entity)):
            adj[entity] = set()
            if entity not in graph:
                continue
            last_hop = {entity}
            for _ in range(1):
                buffer = set()
                for source in last_hop:
                    adj[entity].update(graph[source])
                    buffer.update(graph[source])
                last_hop = buffer
        self.adj = adj
        logger.info(f"[Adjacent Matrix built.]")

    def _build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        self.kg_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.kg_embedding.weight[0], 0)
        logger.debug('[Build embedding]')

    def _build_kg_layer(self):
        # graph encoder
        self.kg_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        if self.pretrain:
            self.kg_encoder.load_state_dict(self.pretrain_data['encoder'])
        # hypergraph convolution
        self.hyper_conv_session = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_knowledge = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        # attention type
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)
        # pooling
        if self.pooling == 'Attn':
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.entity_to_token = nn.Linear(self.kg_emb_dim, self.token_emb_dim, bias=True)
        self.related_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.decoder = TransformerDecoderKG(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        self.copy_proj_1 = nn.Linear(2 * self.token_emb_dim, self.token_emb_dim)
        self.copy_proj_2 = nn.Linear(self.token_emb_dim, self.vocab_size)
        logger.debug('[Build conversation layer]')

    def _get_session_hypergraph(self, session_related_entities):
        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for related_entities in session_related_entities:
            if len(related_entities) == 0:
                continue
            hypergraph_nodes += related_entities
            hypergraph_edges += [hyper_edge_counter] * len(related_entities)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    def _get_knowledge_hypergraph(self, session_related_items):
        related_items_set = set()
        for related_items in session_related_items:
            related_items_set.update(related_items)
        session_related_items = list(related_items_set)

        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for item in session_related_items:
            hypergraph_nodes.append(item)
            hypergraph_edges.append(hyper_edge_counter)
            neighbors = list(self.adj[item])
            hypergraph_nodes += neighbors
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    def _get_knowledge_embedding(self, hypergraph_items, raw_knowledge_embedding):
        knowledge_embedding_list = []
        for item in hypergraph_items:
            sub_graph = [item] + list(self.adj[item])
            sub_graph_embedding = raw_knowledge_embedding[sub_graph]
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)
            knowledge_embedding_list.append(sub_graph_embedding)
        return torch.stack(knowledge_embedding_list, dim=0)

    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    def _attention_and_gating(self, session_embedding, knowledge_embedding, context_embedding):
        related_embedding = torch.cat((session_embedding, knowledge_embedding), dim=0)
        if context_embedding is None:
            if self.pooling == 'Attn':
                user_repr = self.kg_attn_his(related_embedding)
            else:
                assert self.pooling == 'Mean'
                user_repr = torch.mean(related_embedding, dim=0)
        elif self.pooling == 'Attn':
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = self.kg_attn(user_repr)
        else:
            assert self.pooling == 'Mean'
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = torch.mean(attentive_related_embedding, dim=0)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = torch.mean(user_repr, dim=0)
        return user_repr

    def encode_user(self, batch_related_entities, batch_related_items, batch_context_entities, kg_embedding):
        user_repr_list = []
        for session_related_entities, session_related_items, context_entities in zip(batch_related_entities, batch_related_items, batch_context_entities):
            flattened_session_related_items = self.flatten(session_related_items)

            # COLD START
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    user_repr = torch.zeros(self.user_emb_dim, device=self.device)
                elif self.pooling == 'Attn':
                    user_repr = kg_embedding[context_entities]
                    user_repr = self.kg_attn(user_repr)
                else:
                    assert self.pooling == 'Mean'
                    user_repr = kg_embedding[context_entities]
                    user_repr = torch.mean(user_repr, dim=0)
                user_repr_list.append(user_repr)
                continue

            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            session_embedding = self.hyper_conv_session(kg_embedding, session_hyper_edge_index)
            session_embedding = session_embedding[hypergraph_items]
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            raw_knowledge_embedding = self.hyper_conv_knowledge(kg_embedding, knowledge_hyper_edge_index)
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding)
            if len(context_entities) == 0:
                user_repr = self._attention_and_gating(session_embedding, knowledge_embedding, None)
            else:
                context_embedding = kg_embedding[context_entities]
                user_repr = self._attention_and_gating(session_embedding, knowledge_embedding, context_embedding)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)

    def recommend(self, batch, mode):
        related_entities, related_items = batch['related_entities'], batch['related_items']
        context_entities, item = batch['context_entities'], batch['item']
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        extended_items = batch['extended_items']
        for i in range(len(related_items)):
            truncate = min(int(max(2, int(len(related_items[i]) / 4))), len(extended_items[i]))
            if self.extension_strategy == 'Adaptive':
                related_items[i] = related_items[i] + extended_items[i][:truncate]
            else:
                assert self.extension_strategy == 'Random'
                extended_items_sample = random.sample(extended_items[i], truncate)
                related_items[i] = related_items[i] + extended_items_sample

        user_embedding = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, emb_dim)
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        loss = self.rec_loss(scores, item)
        return loss, scores

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def freeze_parameters(self):
        freeze_models = [
            self.kg_embedding,
            self.kg_encoder,
            self.hyper_conv_session,
            self.hyper_conv_knowledge,
            self.item_attn,
            self.rec_bias
        ]
        if self.pooling == "Attn":
            freeze_models.append(self.kg_attn)
            freeze_models.append(self.kg_attn_his)
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    def encode_session(self, batch_related_items, batch_context_entities, kg_embedding):
        """
            Return: session_repr (batch_size, batch_seq_len, token_emb_dim), mask (batch_size, batch_seq_len)
        """
        session_repr_list = []
        for session_related_items, context_entities in zip(batch_related_items, batch_context_entities):
            flattened_session_related_items = self.flatten(session_related_items)

            # COLD START
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    session_repr_list.append(None)
                else:
                    session_repr = kg_embedding[context_entities]
                    session_repr_list.append(session_repr)
                continue

            # TOTAL
            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            session_embedding = self.hyper_conv_session(kg_embedding, session_hyper_edge_index)
            session_embedding = session_embedding[hypergraph_items]
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            raw_knowledge_embedding = self.hyper_conv_knowledge(kg_embedding, knowledge_hyper_edge_index)
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding)
            if len(context_entities) == 0:
                session_repr = torch.cat((session_embedding, knowledge_embedding), dim=0)
                session_repr_list.append(session_repr)
            else:
                context_embedding = kg_embedding[context_entities]
                session_repr = torch.cat((session_embedding, knowledge_embedding, context_embedding), dim=0)
                session_repr_list.append(session_repr)

        batch_seq_len = max([session_repr.size(0) for session_repr in session_repr_list if session_repr is not None])
        mask_list = []
        for i in range(len(session_repr_list)):
            if session_repr_list[i] is None:
                mask_list.append([False] * batch_seq_len)
                zero_repr = torch.zeros((batch_seq_len, self.kg_emb_dim), device=self.device, dtype=torch.float)
                session_repr_list[i] = zero_repr
            else:
                mask_list.append([False] * (batch_seq_len - session_repr_list[i].size(0)) + [True] * session_repr_list[i].size(0))
                zero_repr = torch.zeros((batch_seq_len - session_repr_list[i].size(0), self.kg_emb_dim),
                                        device=self.device, dtype=torch.float)
                session_repr_list[i] = torch.cat((zero_repr, session_repr_list[i]), dim=0)

        session_repr_embedding = torch.stack(session_repr_list, dim=0)
        session_repr_embedding = self.entity_to_token(session_repr_embedding)
        return session_repr_embedding, torch.tensor(mask_list, device=self.device, dtype=torch.bool)

    def decode_forced(self, related_encoder_state, context_encoder_state, session_state, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, related_encoder_state, context_encoder_state, session_state)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

        user_latent = self.entity_to_token(user_embedding)
        user_latent = user_latent.unsqueeze(1).expand(-1, seqlen, -1)
        copy_latent = torch.cat((user_latent, latent), dim=-1)
        copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
        if self.dataset == 'HReDial':
            copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
        sum_logits = token_logits + user_logits + copy_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    def decode_greedy(self, related_encoder_state, context_encoder_state, session_state, user_embedding):
        bsz = context_encoder_state[0].shape[0]
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, related_encoder_state, context_encoder_state, session_state, incr_state)  # incr_state is always None
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

            user_latent = self.entity_to_token(user_embedding)
            user_latent = user_latent.unsqueeze(1).expand(-1, 1, -1)
            copy_latent = torch.cat((user_latent, scores), dim=-1)
            copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
            if self.dataset == 'HReDial':
                copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
            sum_logits = token_logits + user_logits + copy_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def converse(self, batch, mode):
        related_tokens = batch['related_tokens']
        context_tokens = batch['context_tokens']
        related_items = batch['related_items']
        related_entities = batch['related_entities']
        context_entities = batch['context_entities']
        response = batch['response']
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        session_state = self.encode_session(
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, batch_seq_len, token_emb_dim)
        user_embedding = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, emb_dim)
        related_encoder_state = self.related_encoder(related_tokens)
        context_encoder_state = self.context_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(related_encoder_state, context_encoder_state, session_state, user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(related_encoder_state, context_encoder_state, session_state, user_embedding)
            return preds

    def forward(self, batch, mode, stage):
        if len(self.gpu) >= 2:
            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            return self.recommend(batch, mode)