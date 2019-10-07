import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel

import pdb


class QAModel(nn.Module):
    def __init__(self, options):
        super(QAModel, self).__init__()
        self.options = options
        self.bert = BertModel.from_pretrained(
            options.bert_type, cache_dir=options.bert_archive
        )
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        self.qa_outputs = nn.Sequential(
            nn.Linear(options.bert_hidden_size, options.bert_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(options.bert_hidden_size // 2, 2),
        )
        self.fc_yes_no_span = nn.Linear(options.bert_hidden_size, 3, bias=False)

    def construct_binary_mask(self, tensor_in, padding_index=0):
        """For bert. 1 denotes actual data and 0 denotes padding"""
        mask = tensor_in != padding_index
        return mask

    def forward(self, sequences, segment_id):
        sentence_mask = self.construct_binary_mask(sequences)

        paragraph_encoding, pooled_representation = self.bert(
            sequences,
            token_type_ids=segment_id,
            attention_mask=sentence_mask,
            output_all_encoded_layers=False,
        )

        yes_no_span_scores = self.fc_yes_no_span(pooled_representation)

        logits = self.qa_outputs(paragraph_encoding)
        answer_start_scores, answer_end_scores = logits.split(1, dim=-1)
        answer_start_scores = answer_start_scores.squeeze(-1)
        answer_end_scores = answer_end_scores.squeeze(-1)

        return yes_no_span_scores, answer_start_scores, answer_end_scores

