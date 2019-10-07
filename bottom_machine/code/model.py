import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import (
    BertConfig,
    BertPreTrainedModel,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertModel,
)

import pdb


class PassageEncoder(nn.Module):
    def __init__(self, options):
        super(PassageEncoder, self).__init__()
        self.options = options
        self.bert = BertModel.from_pretrained(
            options.bert_type, cache_dir=options.bert_archive
        )
        self.dropout = nn.Dropout(p=options.dropout, inplace=False)
        self.fc_2 = nn.Linear(options.bert_hidden_size, options.transformer_hidden_size)

    def construct_binary_mask(self, tensor_in, padding_index=0):
        """For bert. 1 denotes actual data and 0 denotes padding"""
        mask = tensor_in != padding_index
        return mask

    def forward(self, sequences, segment_id):

        sentence_mask = self.construct_binary_mask(sequences)

        paragraph_encoding, cls_vec = self.bert(
            sequences,
            token_type_ids=segment_id,
            attention_mask=sentence_mask,
            output_all_encoded_layers=False,
        )

        paragraph_encoding = paragraph_encoding[
            :, self.options.max_question_len + 2 :, :
        ]

        paragraph_encoding = self.fc_2(paragraph_encoding)

        return cls_vec, paragraph_encoding


class AnswerTypeClassifier(nn.Module):
    def __init__(self, options):
        super(AnswerTypeClassifier, self).__init__()
        self.options = options
        self.fc_yes_no_span = nn.Linear(options.bert_hidden_size, 3, bias=False)

    def forward(self, features):
        scores = self.fc_yes_no_span(features)
        return scores


class SpanPredictor(BertPreTrainedModel):
    def __init__(self, config, options):
        super(SpanPredictor, self).__init__(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)
        self.options = options
        self.qa_outputs = nn.Sequential(nn.Linear(options.transformer_hidden_size, 2))

    def forward(self, vectors_in, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(
                vectors_in.shape[0], vectors_in.shape[1], device=vectors_in.device
            )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(
            vectors_in, extended_attention_mask, output_all_encoded_layers=False
        )
        encoded_layers = encoded_layers[-1]

        logits = self.qa_outputs(encoded_layers)
        answer_start_scores, answer_end_scores = logits.split(1, dim=-1)
        answer_start_scores = answer_start_scores.squeeze(-1)
        answer_end_scores = answer_end_scores.squeeze(-1)

        return answer_start_scores, answer_end_scores, encoded_layers


class QuestionAnswering(nn.Module):
    def __init__(self, options, config):
        super(QuestionAnswering, self).__init__()
        self.available_gpus = list(range(options.num_para_chunks + 1))
        self.options = options
        self.span_predictor_device = self.available_gpus[0]
        self.passage_encoder = PassageEncoder(options)
        self.answer_type_classifier = AnswerTypeClassifier(options)
        self.span_predictor = SpanPredictor(config, options)
        self.passage_encoder = nn.DataParallel(
            self.passage_encoder,
            device_ids=self.available_gpus[1:],
            output_device=self.available_gpus[0],
        )
        self.fc_sf = nn.Linear(options.transformer_hidden_size * 2, 1)

        self.passage_encoder.to(torch.device("cuda", 1))
        self.answer_type_classifier.to(torch.device("cuda", 0))
        self.span_predictor.to(torch.device("cuda", self.span_predictor_device))
        self.fc_sf.to(torch.device("cuda", self.span_predictor_device))

    def forward(self, data_in):
        """
        data_in:{
            'question_context_sequences_0': ,
            'segment_id_0': ,
            'sentence_start_indices_0': , 
            'sentence_end_indices_0': ,
            'sentence_lengths_0': ,

            ...

            'question_context_sequences_k': ,
            'segment_id_k': ,
            'sentence_start_indices_k': , 
            'sentence_end_indices_k': ,
            'sentence_lengths_k': ,

            supporting_fact:
        }
        """

        # format the batch so that it can be used directly with nn.DataParallel
        new_batch = {"question_context_sequence": [], "segment_id": []}

        for i in range(self.options.num_para_chunks):
            new_batch["question_context_sequence"].append(
                data_in["question_context_sequences_{}".format(i)]
            )
            new_batch["segment_id"].append(data_in["segment_id_{}".format(i)])

        for key in new_batch.keys():
            new_batch[key] = torch.cat(new_batch[key], dim=0)

        cls_vectors, paragraph_encodings = self.passage_encoder(
            sequences=new_batch["question_context_sequence"],
            segment_id=new_batch["segment_id"],
        )

        # Un-format the batch to the original style
        actual_batch_size = data_in["question_context_sequences_0"].shape[0]
        reformatted_batch = {"sequences": [], "cls_vectors": []}

        for i in range(
            0, self.options.num_para_chunks * actual_batch_size, actual_batch_size
        ):
            reformatted_batch["sequences"].append(
                paragraph_encodings[i : i + actual_batch_size, :, :]
            )
            reformatted_batch["cls_vectors"].append(
                cls_vectors[i : i + actual_batch_size, :].unsqueeze(1)
            )

        for key in reformatted_batch.keys():
            reformatted_batch[key] = torch.cat(reformatted_batch[key], dim=1)

        assert (
            len(reformatted_batch["sequences"].shape)
            == len(reformatted_batch["cls_vectors"].shape)
            == 3
        )
        assert (
            reformatted_batch["sequences"].shape[0]
            == reformatted_batch["cls_vectors"].shape[0]
            == actual_batch_size
        )

        cls_summed = reformatted_batch["cls_vectors"].sum(dim=1).squeeze(1)

        assert len(cls_summed.shape) == 2
        assert cls_summed.shape[0] == actual_batch_size

        yes_no_span_scores = self.answer_type_classifier(cls_summed)
        start_scores, end_scores, top_machine_vecs_out = self.span_predictor(
            reformatted_batch["sequences"]
        )

        split_paragraphs = []
        length_of_each_paragraph = int(
            top_machine_vecs_out.shape[1] / self.options.num_para_chunks
        )
        for i in range(0, top_machine_vecs_out.shape[1], length_of_each_paragraph):
            para = top_machine_vecs_out[:, i : i + length_of_each_paragraph, :]
            split_paragraphs.append(para)

        assert len(split_paragraphs) == self.options.num_para_chunks

        sentence_probabilities = []
        for i, para_rep in enumerate(split_paragraphs):
            para_rep_T = para_rep.permute(0, 2, 1)
            start_index = data_in["sentence_start_indices_{}".format(i)].to(
                torch.device("cuda", self.span_predictor_device)
            )
            end_index = data_in["sentence_end_indices_{}".format(i)].to(
                torch.device("cuda", self.span_predictor_device)
            )
            start_vectors = torch.bmm(para_rep_T, start_index.permute(0, 2, 1)).permute(
                0, 2, 1
            )
            end_vectors = torch.bmm(para_rep_T, end_index.permute(0, 2, 1)).permute(
                0, 2, 1
            )

            start_end_concatenated = torch.cat([start_vectors, end_vectors], dim=-1)

            sentence_scores = self.fc_sf(start_end_concatenated).squeeze(dim=-1)
            sent_probs = torch.sigmoid(sentence_scores)
            sentence_probabilities.append(sent_probs)

        sentence_probabilities = torch.cat(sentence_probabilities, dim=1)

        assert len(sentence_probabilities.shape) == 2

        return {
            "yes_no_span_scores": yes_no_span_scores,
            "start_scores": start_scores,
            "end_scores": end_scores,
            "sentence_probabilities": sentence_probabilities,
        }

