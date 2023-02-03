# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class MHIMDataLoader(BaseDataLoader):
    """Dataloader for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab['tok2ind']['__pad__']
        self.start_token_idx = vocab['tok2ind']['__start__']
        self.end_token_idx = vocab['tok2ind']['__end__']
        self.split_token_idx = vocab['tok2ind']['_split_']
        self.related_truncate = opt.get('related_truncate', None)
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = {'related_entities': conv_dict['related_entities'],
                                         'related_items': conv_dict['related_items'],
                                         'extended_items': conv_dict['extended_items'],
                                         'context_entities': conv_dict['context_entities'],
                                         'context_items': conv_dict['context_items'], 'item': movie}
                    augment_dataset.append(augment_conv_dict)

        return augment_dataset

    def rec_batchify(self, batch):
        batch_related_entities = []
        batch_related_items = []
        batch_extended_items = []
        batch_context_entities = []
        batch_context_items = []
        batch_movies = []
        for conv_dict in batch:
            batch_related_entities.append(conv_dict['related_entities'])
            batch_related_items.append(conv_dict['related_items'])
            batch_extended_items.append(conv_dict['extended_items'])
            batch_context_entities.append(conv_dict['context_entities'])
            batch_context_items.append(conv_dict['context_items'])
            batch_movies.append(conv_dict['item'])

        return {
            "related_entities": batch_related_entities,
            "related_items": batch_related_items,
            "extended_items": batch_extended_items,
            "context_entities": batch_context_entities,
            "context_items": batch_context_items,
            "item": torch.tensor(batch_movies, dtype=torch.long)
        }

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_related_tokens = []
        batch_context_tokens = []
        batch_related_entities = []
        batch_related_items = []
        batch_context_entities = []
        batch_response = []
        batch_user_id = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_tokens.append(
                truncate(conv_dict['related_tokens'], self.related_truncate, truncate_tail=False)
            )
            batch_context_tokens.append(
                truncate(merge_utt(
                    conv_dict['context_tokens'],
                    start_token_idx=self.start_token_idx,
                    split_token_idx=self.split_token_idx,
                    final_token_idx=self.end_token_idx
                ), self.context_truncate, truncate_tail=False)
            )
            batch_related_entities.append(conv_dict['related_entities'])
            batch_related_items.append(conv_dict['related_items'])
            batch_context_entities.append(conv_dict['context_entities'])
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))
            batch_user_id.append(conv_dict['user_id'])
            batch_conv_id.append(conv_dict['conv_id'])

        return {
            "related_tokens": padded_tensor(batch_related_tokens, self.pad_token_idx, pad_tail=False),
            "context_tokens": padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
            "related_entities": batch_related_entities,
            "related_items": batch_related_items,
            "context_entities": batch_context_entities,
            "response": padded_tensor(batch_response, self.pad_token_idx),
            "user_id": batch_user_id,
            "conv_id": batch_conv_id
        }

    def policy_batchify(self, *args, **kwargs):
        pass
