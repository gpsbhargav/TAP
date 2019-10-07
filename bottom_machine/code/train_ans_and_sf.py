import os
import time
import glob
import re
import string
import json
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertConfig

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import utils
from model import QuestionAnswering
import options

import argparse

import pdb

VERY_NEGATIVE_NUMBER = -1e-30


def multi_span_loss(gt_start_indices, gt_end_indices, pred_start_logits,pred_end_logits):
    def gather_logsumexp_true_indices(start_indices, end_indices, start_scores, end_scores):
        start_logsumexp = []
        end_logsumexp = []
        for i in range(len(start_indices)):
            s_scores = []
            e_scores = []
            for idx in start_indices[i]:
                if(idx == -1):
                    break
                s_scores.append(start_scores[i][idx].unsqueeze(0))
            for idx in end_indices[i]:
                if(idx == -1):
                    break
                e_scores.append(end_scores[i][idx].unsqueeze(0))
            
            assert(len(s_scores) == len(e_scores))
            
            if(len(s_scores) == 0):
                s_scores.append(start_scores[i][0].unsqueeze(0))
                e_scores.append(end_scores[i][0].unsqueeze(0))

            s_scores = torch.cat(s_scores,dim=-1)
            e_scores = torch.cat(e_scores,dim=-1)

            s_lse = torch.logsumexp(s_scores,dim=-1)
            e_lse = torch.logsumexp(e_scores,dim=-1)
            start_logsumexp.append(s_lse.unsqueeze(0))
            end_logsumexp.append(e_lse.unsqueeze(0))

        start_logsumexp = torch.cat(start_logsumexp, dim=-1)
        end_logsumexp = torch.cat(end_logsumexp, dim=-1)

        return start_logsumexp, end_logsumexp

    start_true_logsumexp, end_true_logsumexp = gather_logsumexp_true_indices(start_indices = gt_start_indices, end_indices = gt_end_indices, start_scores = pred_start_logits, end_scores = pred_end_logits)
    
    start_all_logsumexp = torch.logsumexp(pred_start_logits,dim=-1)
    end_all_logsumexp = torch.logsumexp(pred_end_logits,dim=-1)
    
    start_loss = -(start_true_logsumexp - start_all_logsumexp)
    end_loss = -(end_true_logsumexp - end_all_logsumexp)
    
    total_loss =  start_loss + end_loss
    return total_loss


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class PredictedSpanFormatter:
    def __init__(self, max_answer_length=15):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_answer_length = max_answer_length  # options.max_answer_length

    def un_tokenize(self, ids, tokens_to_text_mapping, bert_tokenizer):
        out_list = []
        start = 0
        end = start
        while (start < len(ids)) and (end < len(ids)):
            i = len(ids)
            decoded_anything = False
            while (decoded_anything == False) and (i > start):
                if (tuple(ids[start:i]) in tokens_to_text_mapping.keys()):
                    out_list.append(tokens_to_text_mapping[tuple(
                        ids[start:i])])
                    decoded_anything = True
                else:
                    i -= 1
            if (decoded_anything == False):
                out_list.append(
                    bert_tokenizer.convert_ids_to_tokens([ids[start]])[0])
                start += 1
                end = start
            else:
                start = i
                end = i
        return " ".join(out_list)

    def is_word_split(self, word):
        if (len(word) < 2):
            return False
        else:
            return (word[0] == '#' and word[1] == '#')

    def combine_word_pieces(self, sentence):
        # the first word cant start with '##'
        out_tokens = []
        for token in sentence:
            if (not self.is_word_split(token)):
                out_tokens.append((token))
            else:
                out_tokens[-1] += token[2:]
        return out_tokens

    def convert_indices_to_text(self, sentence, start, end,
                                tokens_to_text_mapping):
        ''' (sentence, [10, 12]) --> ['runn', '##ing', 'race'] --> ['running', 'race']
        --> "running race" '''
        text = self.tokenizer.convert_ids_to_tokens(sentence)
        true_start = start
        if (self.is_word_split(text[start])):
            for i in range(1, start):
                if (not self.is_word_split(text[start - i])):
                    true_start = start - i
                    break

        true_end = end
        for i in range(end + 1, len(sentence)):
            if (not self.is_word_split(text[i])):
                true_end = i - 1
                break

        proper_text = self.un_tokenize(sentence[true_start:true_end + 1],
                                       tokens_to_text_mapping, self.tokenizer)
        #         proper_text = " ".join(text[true_start:true_end+1]).replace('  ##','').replace(' ##','')
        return proper_text

    def find_most_confident_span(self, start_scores, end_scores):
        ''' 
        Inputs: masked start_scores and end_scores of a single example
        Output: (i,j) pairs having highest Pr(i) + Pr(j)
        '''
        assert (len(start_scores) == len(end_scores))
        best_start = 0
        best_stop = 0
        best_confidence = -1e100
        for i in range(len(start_scores)):
            for j in range(
                    min(len(end_scores), i + self.max_answer_length) - 1,
                    i - 1, -1):
                if (math.log(start_scores[i]) + math.log(end_scores[j]) >
                        best_confidence):
                    best_start = i
                    best_stop = j
                    #                     best_confidence = start_scores[i] + end_scores[j]
                    best_confidence = math.log(start_scores[i]) + math.log(
                        end_scores[j])
        return best_start, best_stop

    def find_top_n_confident_spans(self, start_scores, end_scores, n):
        ''' 
        Inputs: masked start_scores and end_scores of a single example
        Output: (i,j) n pairs having highest Pr(i) + Pr(j)
        '''
        assert (len(start_scores) == len(end_scores))
        scores = []
        for i in range(len(start_scores)):
            for j in range(
                    min(len(end_scores) - 1, i + self.max_answer_length - 1),
                    i - 1, -1):
                s = math.log(start_scores[i]) + math.log(end_scores[j])
                scores.append([s, i, j, start_scores[i], end_scores[j]])
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:n]

    def format_prediction(self,
                          yes_no_span,
                          start_scores,
                          end_scores,
                          sequences,
                          tokens_to_text_mappings,
                          question_ids,
                          max_question_len,
                          official_evalutation=True):
        '''
        input: all numpy arrays
        output: {"question_id": answer_string}
        '''
        answers = {}

        assert (len(yes_no_span) == len(start_scores) == len(end_scores) == len(sequences))

        for i in range(len(yes_no_span)):
            if (official_evalutation):
                yns = yes_no_span[i].argmax(axis=-1)
                if (yns == 0):
                    answers[question_ids[i]] = "yes"
                    continue
                elif (yns == 1):
                    answers[question_ids[i]] = "no"
                    continue

            start, end = self.find_most_confident_span(start_scores[i],
                                                       end_scores[i])

            sequence_chunks_concatenated = []
            for seq in sequences[i]:
                sequence_chunks_concatenated += seq[max_question_len + 2:]

            ans = self.convert_indices_to_text(sequence_chunks_concatenated,
                                               start, end,
                                               tokens_to_text_mappings[i])
            answers[question_ids[i]] = ans

        assert (len(answers) == len(sequences))

        return answers


class SupportingFactFormatter:
    '''
    inputs: 
    - a binary array for each question. It will have 1 if the corresponding sentence is a supporting fact 0 otherwise.
    - question id
    - names of paragraphs in the context
    - which paragraph is in which chunk
    - number of sentences in each paragraph
    
    output:
    A list like this
    [['Bridgeport, Connecticut', 5], ['Brookhaven National Laboratory', 1]]
    '''

    def __init__(self, num_chunks, num_sentences_per_chunk):
        self.num_chunks = num_chunks
        self.num_sentences_per_chunk = num_sentences_per_chunk

    def find_all_indices(self, the_array, the_value):
        assert (len(the_array.shape) == 1)
        return list(np.where(the_array == the_value)[0])

    def find_paragraph_and_sentence_index(self, sent_index,
                                          paragraph_chunk_indices,
                                          num_sentences_in_paragraphs):
        chunk_index = sent_index // self.num_sentences_per_chunk
        assert (chunk_index < self.num_chunks)
        sent_index = sent_index - (chunk_index * self.num_sentences_per_chunk)
        num_sents_cum_sum = 0
        para_index = -1
        actual_sentence_index = -1
        for p_index in paragraph_chunk_indices[chunk_index]:
            if (num_sents_cum_sum <= sent_index <
                    num_sents_cum_sum + num_sentences_in_paragraphs[p_index]):
                para_index = p_index
                actual_sentence_index = sent_index - num_sents_cum_sum
                break
            else:
                num_sents_cum_sum += num_sentences_in_paragraphs[p_index]
        return para_index, actual_sentence_index

    def find_paragraph_name(self, para_index, paragraph_names):
        assert (0 <= para_index)
        return paragraph_names[para_index]

    def format_supporting_facts(self, predictions, question_ids,
                                paragraph_names, paragraph_chunk_indices,
                                num_sentences_in_paragraphs):
        assert (len(predictions) == len(question_ids) == len(paragraph_names)
                == len(paragraph_chunk_indices) ==
                len(num_sentences_in_paragraphs))

        out_records = {}
        for i, pred_row in enumerate(predictions):
            indices_of_sf = self.find_all_indices(the_array=pred_row,
                                                  the_value=1)
            formatted_sf_list = []
            for sf_idx in indices_of_sf:
                para_idx, sentence_idx = self.find_paragraph_and_sentence_index(
                    sent_index=sf_idx,
                    paragraph_chunk_indices=paragraph_chunk_indices[i],
                    num_sentences_in_paragraphs=num_sentences_in_paragraphs[i])
                if (para_idx < 0 or sentence_idx < 0):
                    continue
                para_name = self.find_paragraph_name(
                    para_index=para_idx, paragraph_names=paragraph_names[i])
                formatted_sf_list.append([para_name, sentence_idx])
            out_records[question_ids[i]] = formatted_sf_list

        return out_records


class Evaluator:
    '''Adapted from the official evaluation script'''

    def compute_yes_no_span_em(self, gt, pred):
        pred_classes = pred.argmax(axis=-1)
        num_correct = 0
        for i in range(len(pred)):
            if(pred_classes[i] == gt[i]):
                num_correct += 1
        return num_correct/len(pred)

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if normalized_prediction in [
                'yes', 'no', 'noanswer'
        ] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in [
                'yes', 'no', 'noanswer'
        ] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def exact_match_score(self, prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(
            ground_truth))

    def update_answer(self, metrics, prediction, gold):
        em = self.exact_match_score(prediction, gold)
        f1, prec, recall = self.f1_score(prediction, gold)
        metrics['em'] += float(em)
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall
        return em, prec, recall

    def update_sp(self, metrics, prediction, gold):
        cur_sp_pred = set(map(tuple, prediction))
        gold_sp_pred = set(map(tuple, gold))
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        metrics['sp_em'] += em
        metrics['sp_f1'] += f1
        metrics['sp_prec'] += prec
        metrics['sp_recall'] += recall
        return em, prec, recall

    def eval(self, prediction, gold):
        metrics = {
            'em': 0,
            'f1': 0,
            'prec': 0,
            'recall': 0,
            'sp_em': 0,
            'sp_f1': 0,
            'sp_prec': 0,
            'sp_recall': 0,
            'joint_em': 0,
            'joint_f1': 0,
            'joint_prec': 0,
            'joint_recall': 0
        }
        for dp in gold:
            cur_id = dp['_id']
            can_eval_joint = True
            if cur_id not in prediction['answer']:
                print('missing answer {}'.format(cur_id))
                can_eval_joint = False
            else:
                em, prec, recall = self.update_answer(
                    metrics, prediction['answer'][cur_id], dp['answer'])
            if cur_id not in prediction['sp']:
                print('missing sp fact {}'.format(cur_id))
                can_eval_joint = False
            else:
                sp_em, sp_prec, sp_recall = self.update_sp(
                    metrics, prediction['sp'][cur_id], dp['supporting_facts'])

            if can_eval_joint:
                joint_prec = prec * sp_prec
                joint_recall = recall * sp_recall
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec +
                                                                joint_recall)
                else:
                    joint_f1 = 0.
                joint_em = em * sp_em

                metrics['joint_em'] += joint_em
                metrics['joint_f1'] += joint_f1
                metrics['joint_prec'] += joint_prec
                metrics['joint_recall'] += joint_recall

        N = len(gold)
        for k in metrics.keys():
            metrics[k] /= N
        
        return metrics


class QADataset(Dataset):
    def __init__(self, data, options=None):
        self.data = data
        assert (data['max_seq_len'] == options.max_seq_len)
        self.max_seq_len = data['max_seq_len']
        self.max_question_len = data['max_question_len']
        self.max_num_sentences_per_chunk = data['max_num_sentences_per_chunk']
        self.num_chunks = data['num_chunks']
        self.max_context_len_per_chunk = self.max_seq_len - self.max_question_len - 2
        self.max_num_answers = 10

    def __len__(self):
        return len(self.data['question_context_sequences'])

    def get_supporting_fact_related_things(self, index):
        out_dict = {}
        for i in range(self.num_chunks):
            sequence = self.data['question_context_sequences'][index][i]
            sentence_start_indices = self.data[
                'sentence_start_indices'][index][i]
            sentence_end_indices = self.data[
                'sentence_end_indices'][index][i]

            supporting_fact = self.data['supporting_facts_expanded'][index]

            start_index_matrix = []
            end_index_matrix = []
            for j in range(len(sentence_start_indices)):
                start_indicator_vector = [0] * self.max_context_len_per_chunk
                end_indicator_vector = [0] * self.max_context_len_per_chunk
                start_indicator_vector[sentence_start_indices[j]] = 1
                end_indicator_vector[sentence_end_indices[j]] = 1
                start_index_matrix.append(start_indicator_vector)
                end_index_matrix.append(end_indicator_vector)

            if (self.max_num_sentences_per_chunk - len(sentence_start_indices) > 0):
                fake_sentence_start_and_end = [0] * self.max_context_len_per_chunk
                index_of_pad = sequence[self.max_question_len + 2:].index(0)  # pad_index is 0
                fake_sentence_start_and_end[index_of_pad] = 1
                for j in range(self.max_num_sentences_per_chunk -
                               len(sentence_start_indices)):
                    start_index_matrix.append(fake_sentence_start_and_end)
                    end_index_matrix.append(fake_sentence_start_and_end)

            sentence_lengths = []
            for j in range(len(sentence_start_indices)):
                l = (sentence_end_indices[j] + 1) - sentence_start_indices[j]
                sentence_lengths.append(l)

            sentence_lengths += [0] * (self.max_num_sentences_per_chunk -
                                       len(sentence_lengths))

            assert (len(start_index_matrix) == self.max_num_sentences_per_chunk)
            assert (len(end_index_matrix) == self.max_num_sentences_per_chunk)

            out_dict['sentence_start_indices_{}'.format(i)] = torch.tensor(start_index_matrix, dtype=torch.float32)
            out_dict['sentence_end_indices_{}'.format(i)] = torch.tensor(end_index_matrix,
                                         dtype=torch.float32)
            out_dict['supporting_fact'] = torch.tensor(supporting_fact, dtype=torch.float32)
            out_dict['sentence_lengths_{}'.format(i)] = torch.tensor(sentence_lengths)

        return out_dict

    def get_span_related_things(self, index):
        out_dict = {}

        for i in range(self.num_chunks):
            out_dict['question_context_sequences_{}'.format(
                i)] = torch.tensor(self.data['question_context_sequences'][index][i])
            out_dict['segment_id_{}'.format(i)] = torch.tensor(
                self.data['segment_id'])

            assert (len(self.data['answer_start_indices'][index]) == len(
                self.data['answer_end_indices'][index]))

        start_indices = torch.zeros(self.max_context_len_per_chunk*self.num_chunks, dtype=torch.float32)
        start_indices[self.data['answer_start_indices'][index]] = 1.0

        end_indices = torch.zeros(self.max_context_len_per_chunk*self.num_chunks, dtype=torch.float32)
        end_indices[self.data['answer_end_indices'][index]] = 1.0

        yes_no_span = torch.tensor(self.data['yes_no_span'][index])

        if (self.data['yes_no_span'][index] == 2):
            loss_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            loss_mask = torch.tensor(0.0, dtype=torch.float32)

        if (len(self.data['answer_start_indices'][index]) <= 0):
            start_indices = [0]
            end_indices = [0]
        else:
            start_indices = self.data['answer_start_indices'][index]
            end_indices = self.data['answer_end_indices'][index]
        
        start_indices = start_indices[:self.max_num_answers]
        start_indices += [-1] * (self.max_num_answers - len(start_indices))

        end_indices = end_indices[:self.max_num_answers]
        end_indices += [-1] * (self.max_num_answers - len(end_indices))

        assert(len(start_indices) == len(end_indices))

        question_index = torch.tensor(self.data['question_indices'][index])

        out_dict['answer_start_indices'] = torch.tensor(start_indices)
        out_dict['answer_end_indices'] = torch.tensor(end_indices)
        out_dict['yes_no_span'] = yes_no_span
        out_dict['loss_mask'] = loss_mask
        out_dict['question_indices'] = question_index
        out_dict['example_indices'] = torch.tensor(index)

        return out_dict
    
    def __getitem__(self, index):
        out_dict = self.get_span_related_things(index)
        d2 = self.get_supporting_fact_related_things(index)

        for key, value in d2.items():
            out_dict[key] = value
        
        return out_dict


def prepare_gt_for_question_indices(data, question_indices=None):
    '''
    Prepares ground truth in a way that can be fed to the evaluation script.
    '''
    if(question_indices is None):
        question_indices = list(range(len(data['question_ids'])))
    out_records = []
    for i in question_indices:
        record = {}
        record['_id'] = data['question_ids'][i]
        record['answer'] = data['answers_string'][i]
        record['supporting_facts'] = data['supporting_facts_raw'][i]
        out_records.append(record)
    return out_records



def direct_sf_evaluate(gt, pred):
    assert(len(gt) == len(pred))
    total_size = len(pred)
    assert(len(gt) != 0)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_correct = 0
    for i in range(total_size):
        if(np.array_equal(gt[i], pred[i])):
            total_correct += 1
        p = precision_score(gt[i], pred[i],average="binary")
        r = recall_score(gt[i], pred[i],average="binary")
        total_precision += p
        total_recall += r
        total_f1 += 2*(p*r)/(p+r) if (p+r)>0 else 0
    return {"precision":total_precision/total_size, "recall":total_recall/total_size, 
            "f1":total_f1/total_size, "em":total_correct/total_size}


options = options.StrongSupervisionOptions()

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--short_run', action='store_true')
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--use_small_dataset', action='store_true')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--overwrite_save', action='store_true')
parser.add_argument('--sf_weight', type=int, default=4)
command_line_args = parser.parse_args()

options.resume_training = command_line_args.resume_training
options.dev_only = command_line_args.eval_only
options.learning_rate = command_line_args.lr
options.epochs = command_line_args.epochs

options.set_experiment_name(command_line_args.name)
if(command_line_args.use_small_dataset):
    options.set_use_small_dataset()
if(command_line_args.short_run):
    options.set_debugging_short_run()


# create experiment directory 
if not os.path.exists(options.save_path):
    os.mkdir(options.save_path)
else:
    assert(command_line_args.overwrite_save == True or options.dev_only == True or options.resume_training == True), "Experiment directory already exists."

span_formatter = PredictedSpanFormatter()
sf_formatter = SupportingFactFormatter(options.num_para_chunks, options.num_sentences_per_chunk)
evaluator = Evaluator()

print("Reading data pickles")

train_data = utils.unpickler(options.data_pkl_path, options.train_pkl_name)
dev_data = utils.unpickler(options.data_pkl_path, options.dev_pkl_name)

train_dataset = QADataset(train_data, options)
dev_dataset = QADataset(dev_data, options)


train_data_loader = DataLoader(train_dataset,
                               batch_size=options.batch_size,
                               shuffle=True,
                               sampler=None,
                               batch_sampler=None,
                               num_workers=8,
                               pin_memory=False,
                               drop_last=False,
                               timeout=0,
                               worker_init_fn=None)

dev_data_loader = DataLoader(dev_dataset,
                             batch_size=options.dev_batch_size,
                             shuffle=False,
                             sampler=None,
                             batch_sampler=None,
                             num_workers=1,
                             pin_memory=False,
                             drop_last=False,
                             timeout=8,
                             worker_init_fn=None)

print("Building model")

transformer_config = BertConfig(vocab_size_or_config_json_file=30522,hidden_size=options.transformer_hidden_size, num_hidden_layers=options.num_transformer_layers, num_attention_heads=8, intermediate_size=2048)

model = QuestionAnswering(options,transformer_config)

# print("===============================")
# print("Model:")
# print(model)
# print("===============================")


criterion_yes_no_span = nn.CrossEntropyLoss()
criterion_sf = nn.BCELoss()

# Prepare optimizer
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{
    'params':
    [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay':
    0.01
}, {
    'params':
    [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay':
    0.0
}]

num_train_steps = int((len(train_dataset) / options.batch_size /
                       options.gradient_accumulation_steps) * options.epochs)

t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=options.learning_rate,
                     warmup=options.warmup_proportion,
                     t_total=t_total)

routine_log_template = 'Time:{:.1f}, Epoch:{}/{}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, Yes_No_Span:{:.2f}'

print("Training data size:{}".format(len(train_dataset)))
print("Dev data size:{}".format(len(dev_dataset)))

total_loss_since_last_time = 0
num_evaluations_since_last_best_dev_acc = 0
dev_predictions_best_model = None
stop_training_flag = False

iterations = 0
best_dev_f1 = -1
start_epoch = 0

if options.resume_training or options.dev_only:
    if os.path.isfile(os.path.join(options.save_path,
                                   options.checkpoint_name)):
        print("=> loading checkpoint")
        checkpoint = torch.load(
            os.path.join(options.save_path, options.checkpoint_name), map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_dev_f1 = checkpoint['best_acc']
        iterations = checkpoint['iteration']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint. Resuming epoch {}, iteration {}".format(
            checkpoint['epoch'] + 1, checkpoint['iteration']))

torch.cuda.empty_cache()

if (options.dev_only):
    best_dev_f1 = -1


if not options.dev_only:
    print("Training now")

if (options.debugging_short_run):
    print("This is a short run")

start = time.time()

for epoch in range(start_epoch, options.epochs):
    for batch_idx, train_batch in enumerate(train_data_loader):
        if (options.dev_only):
            break

        if (iterations > num_train_steps):
            print("Reached maximum number of iterations")
            break

        if (options.debugging_short_run):
            if (batch_idx == options.debugging_num_iterations + 1):
                break

        model.train()
        optimizer.zero_grad()

        answer = model(train_batch)

        # move the ground truth to the same devices where the predictions lie
        train_batch['answer_start_indices'] = train_batch['answer_start_indices'].cuda(answer['start_scores'].get_device()) 
        train_batch['answer_end_indices'] = train_batch['answer_end_indices'].cuda(answer['end_scores'].get_device()) 
        train_batch['yes_no_span'] = train_batch['yes_no_span'].cuda(answer['yes_no_span_scores'].get_device()) 
        train_batch['supporting_fact'] = train_batch['supporting_fact'].cuda(answer['sentence_probabilities'].get_device()) 


        loss_span = multi_span_loss(gt_start_indices=train_batch['answer_start_indices'], gt_end_indices=train_batch['answer_end_indices'], pred_start_logits=answer['start_scores'],pred_end_logits=answer['end_scores'])

        train_batch['loss_mask'] = train_batch['loss_mask'].cuda(loss_span.get_device())
        loss_span = (loss_span * train_batch['loss_mask']).mean()
        
        loss_yes_no_span = criterion_yes_no_span(answer['yes_no_span_scores'], train_batch['yes_no_span'])

        total_loss = (options.loss_weight_span * loss_span) +(options.loss_weight_yes_no_span * loss_yes_no_span) 

        loss_sf = criterion_sf(answer['sentence_probabilities'],train_batch['supporting_fact'])
        total_loss += command_line_args.sf_weight * loss_sf 

        total_loss_since_last_time += total_loss.item()

        total_loss.backward()

        lr_this_step = options.learning_rate * \
            warmup_linear(iterations/t_total, options.warmup_proportion)

        if (lr_this_step < 0):
            print("Learning rate < 0. Stopping training")
            break

        assert (lr_this_step >= 0)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step

        optimizer.step()

        iterations += 1

        if (torch.isnan(total_loss).item()):
            print("Loss became nan in iteration {}. Training stopped".format(
                iterations))
            stop_training_flag = True
            break
        elif (total_loss.item() < 0.0000000000001):
            print("Loss is too low. Stopping training")
            stop_training_flag = True
            break

        if iterations % options.log_every == 0:

            id_to_token_maps = []
            train_question_ids = []
            train_paragraph_names = []
            paragraph_chunk_indices = []
            num_sentences_in_paragraphs = []
            train_question_context_sequences = []
            for index in train_batch['example_indices']:
                mapping = train_data["ids_to_word_mappings"][index]
                id_to_token_maps.append(mapping)
                train_question_ids.append(train_data["question_ids"][index])
                train_paragraph_names.append(train_data["paragraph_names"][index])
                paragraph_chunk_indices.append(train_data["paragraph_chunk_indices"][index])
                num_sentences_in_paragraphs.append(train_data["num_sentences_in_paragraphs"][index])
                train_question_context_sequences.append(train_data["question_context_sequences"][index])

            ans_formatted = span_formatter.format_prediction(
                yes_no_span=torch.softmax(answer['yes_no_span_scores'],
                                          dim=-1).detach().cpu().numpy(),
                start_scores=torch.softmax(answer['start_scores'],
                                           dim=-1).detach().cpu().numpy(),
                end_scores=torch.softmax(answer['end_scores'],
                                         dim=-1).detach().cpu().numpy(),
                sequences=train_question_context_sequences,
                tokens_to_text_mappings=id_to_token_maps,
                question_ids = train_question_ids,
                max_question_len = options.max_question_len)

            gold_formatted = prepare_gt_for_question_indices(data=train_data, question_indices=train_batch['example_indices'].detach().cpu().numpy())
            
            best_train_metrics = None
            best_train_threshold = 0.5
            best_train_predictions = None
            for threshold in np.linspace(0.1,0.9,5):
                thresholded_sf = answer['sentence_probabilities'].detach().cpu() > threshold
                sf_formatted = sf_formatter.format_supporting_facts(
                    predictions = thresholded_sf.numpy(), 
                    question_ids = train_question_ids,
                    paragraph_names = train_paragraph_names, 
                    paragraph_chunk_indices = paragraph_chunk_indices,
                    num_sentences_in_paragraphs = num_sentences_in_paragraphs
                )
                train_predictions_formatted = {'answer':ans_formatted, 'sp':sf_formatted}
                train_metrics = evaluator.eval(train_predictions_formatted,gold_formatted)
                if(best_train_metrics is None):
                    best_train_metrics = train_metrics
                    best_train_threshold = threshold
                    best_train_predictions = train_predictions_formatted
                elif(train_metrics["joint_f1"] >= best_train_metrics["joint_f1"]):
                    best_train_metrics = train_metrics
                    best_train_threshold = threshold
                    best_train_predictions = train_predictions_formatted

            train_yes_no_span_em = evaluator.compute_yes_no_span_em(
                gt=train_batch['yes_no_span'].detach().cpu().numpy(),
                pred=answer['yes_no_span_scores'].detach().cpu().numpy())

            avg_loss = total_loss_since_last_time / options.log_every
            total_loss_since_last_time = 0


            
            
            print('- - - - - - - - - - - - - - - - - - - -')
            print(
                routine_log_template.format(
                    time.time() - start, epoch + 1, options.epochs, iterations,
                    avg_loss, total_loss.item(), train_yes_no_span_em))
            print(best_train_metrics)
            print("Best sf threshold: {}".format(best_train_threshold))

            if iterations % options.save_every == 0:
                snapshot_prefix = os.path.join(options.save_path,
                                               options.checkpoint_name)
                snapshot_path = snapshot_prefix
                state = {
                    'epoch': epoch,
                    'iteration': iterations,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_dev_f1,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

    if (stop_training_flag == True):
        break

    print("Evaluating on dev set")

    # switch model to evaluation mode
    model.eval()

    dev_all_masked_start = []
    dev_all_masked_end = []
    dev_all_yes_no_span = []
    dev_all_sf = []
     
    with torch.no_grad():
        for dev_batch_idx, dev_batch in enumerate(dev_data_loader):

            if (options.debugging_short_run):
                if (options.debugging_num_dev_iterations < dev_batch_idx):
                    break

            dev_answer = model(dev_batch)

            dev_masked_start = torch.softmax(dev_answer['start_scores'], dim=-1)
            dev_masked_end = torch.softmax(dev_answer['end_scores'], dim=-1)
            dev_yns = torch.softmax(dev_answer['yes_no_span_scores'],
                                    dim=-1).detach().cpu().numpy()
            dev_sf =   dev_answer['sentence_probabilities'].detach().cpu().numpy()                     
            dev_all_masked_start.append(
                dev_masked_start.detach().cpu().numpy())
            dev_all_masked_end.append(dev_masked_end.detach().cpu().numpy())
            dev_all_yes_no_span.append(dev_yns)
            dev_all_sf.append(dev_sf)

    dev_all_masked_start = np.concatenate(dev_all_masked_start, axis=0)
    dev_all_masked_end = np.concatenate(dev_all_masked_end, axis=0)
    dev_all_yes_no_span = np.concatenate(dev_all_yes_no_span, axis=0)
    dev_all_sf = np.concatenate(dev_all_sf, axis=0)
    
    dev_ans_formatted = span_formatter.format_prediction(
                yes_no_span= dev_all_yes_no_span,
                start_scores= dev_all_masked_start,
                end_scores= dev_all_masked_end,
                sequences= dev_data['question_context_sequences'],
                tokens_to_text_mappings=dev_data['ids_to_word_mappings'],
                question_ids = dev_data['question_ids'],
                max_question_len = options.max_question_len)
    
    
    dev_gold_formatted = prepare_gt_for_question_indices(data=dev_data)
    
    best_dev_metrics = None
    best_dev_threshold = 0.5
    best_dev_predictions = None
    for threshold in np.linspace(0.1,0.9,9):
        dev_thresholded_sf = (torch.tensor(dev_all_sf) > threshold).numpy()
        dev_sf_formatted = sf_formatter.format_supporting_facts(
                predictions = dev_thresholded_sf, 
                question_ids = dev_data['question_ids'],
                paragraph_names = dev_data["paragraph_names"], 
                paragraph_chunk_indices = dev_data["paragraph_chunk_indices"],
                num_sentences_in_paragraphs = dev_data["num_sentences_in_paragraphs"]
            )
        dev_predictions_formatted = {'answer':dev_ans_formatted, 'sp':dev_sf_formatted}
        
        dev_metrics = evaluator.eval(dev_predictions_formatted,dev_gold_formatted)

        if(best_dev_metrics is None):
            best_dev_metrics = dev_metrics
            best_dev_threshold = threshold
            best_dev_predictions = dev_predictions_formatted
        elif(dev_metrics["joint_f1"] >= best_dev_metrics["joint_f1"]):
            best_dev_metrics = dev_metrics
            best_dev_threshold = threshold
            best_dev_predictions = dev_predictions_formatted
    
    dev_f1 = best_dev_metrics["joint_f1"]


    dev_yes_no_span_em = evaluator.compute_yes_no_span_em(
        gt=dev_data['yes_no_span'], pred=dev_all_yes_no_span)
    

    print('==================================')
    print("Dev set:")
    print("Yes/no/span EM: {}".format(dev_yes_no_span_em))
    print("Official evaluation metrics:")
    print(best_dev_metrics)
    print("Best threshold: {}".format(best_dev_threshold))
    print('==================================')

    with open(options.acc_log_file, "a") as acc_log_file_handle:
        dev_all_metrics = best_dev_metrics
        dev_all_metrics["yes_no_span_em"] = dev_yes_no_span_em
        acc_log_file_handle.write(
            "\n --{}-- \n Epoch :{} , metrics: {}".format(
                options.experiment_name, epoch, json.dumps(dev_all_metrics)))

    # update best valiation set accuracy
    if dev_f1 > best_dev_f1:

        dev_predictions_best_model = [
            dev_all_yes_no_span, dev_all_masked_start, dev_all_masked_end, dev_all_sf
        ]

        num_evaluations_since_last_best_dev_acc = 0

        # found a model with better validation set accuracy

        best_dev_f1 = dev_f1
        snapshot_prefix = os.path.join(options.save_path, 'best_snapshot')
        snapshot_path = snapshot_prefix + \
            '_dev_f1_{}_iter_{}_model.pt'.format(dev_f1, iterations)

        # save model, delete previous 'best_snapshot' files
        state = {
            'epoch': epoch,
            'iteration': iterations,
            'model_state_dict': model.state_dict(),
            'best_acc': best_dev_f1,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)

        # save best predictions
        utils.pickler(options.save_path, options.predictions_pkl_name,
                      dev_predictions_best_model)
    else:
        num_evaluations_since_last_best_dev_acc += 1

    if (num_evaluations_since_last_best_dev_acc >
            options.early_stopping_patience):
        print(
            "Training stopped because dev acc hasn't increased in {} epochs.".
            format(options.early_stopping_patience))
        print("Best dev set accuracy = {}".format(best_dev_metrics))
        break

    if (options.debugging_short_run):
        print("Short run completed")
        break

    if (options.dev_only):
        break
