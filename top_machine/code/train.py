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

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import utils
from model import QAModel
import options

import pdb


VERY_NEGATIVE_NUMBER = -1e-30


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class PredictionFormatter:
    def __init__(self, options):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir=options.bert_archive
        )
        self.max_answer_length = options.max_answer_length

    def un_tokenize(self, ids, tokens_to_text_mapping, bert_tokenizer):
        out_list = []
        start = 0
        end = start
        while (start < len(ids)) and (end < len(ids)):
            i = len(ids)
            decoded_anything = False
            while (decoded_anything == False) and (i > start):
                if tuple(ids[start:i]) in tokens_to_text_mapping.keys():
                    out_list.append(tokens_to_text_mapping[tuple(ids[start:i])])
                    decoded_anything = True
                else:
                    i -= 1
            if decoded_anything == False:
                out_list.append(bert_tokenizer.convert_ids_to_tokens([ids[start]])[0])
                start += 1
                end = start
            else:
                start = i
                end = i
        return " ".join(out_list)

    def is_word_split(self, word):
        if len(word) < 2:
            return False
        else:
            return word[0] == "#" and word[1] == "#"

    def combine_word_pieces(self, sentence):
        # the first word cant start with '##'
        out_tokens = []
        for token in sentence:
            if not self.is_word_split(token):
                out_tokens.append((token))
            else:
                out_tokens[-1] += token[2:]
        return out_tokens

    def convert_indices_to_text(self, sentence, start, end, tokens_to_text_mapping):
        """ (sentence, [10, 12]) --> ['runn', '##ing', 'race'] --> ['running', 'race']
        --> "running race" """
        text = self.tokenizer.convert_ids_to_tokens(sentence)
        true_start = start
        if self.is_word_split(text[start]):
            for i in range(1, start):
                if not self.is_word_split(text[start - i]):
                    true_start = start - i
                    break

        true_end = end
        for i in range(end + 1, len(sentence)):
            if not self.is_word_split(text[i]):
                true_end = i - 1
                break

        proper_text = self.un_tokenize(
            sentence[true_start : true_end + 1], tokens_to_text_mapping, self.tokenizer
        )
        return proper_text

    #     def find_most_confident_span(self, start_scores, end_scores):
    #         '''
    #         Inputs: masked start_scores and end_scores of a single example
    #         Output: (i,j) pairs having highest Pr(i) + Pr(j)
    #         '''
    #         assert(len(start_scores) == len(end_scores))
    #         best_start = 0
    #         best_stop = 0
    #         best_confidence = 0
    #         for i in range(len(start_scores)):
    #             for j in range(i, min(len(end_scores), i + self.max_answer_length)):
    #                 if(start_scores[i] + end_scores[j] > best_confidence):
    #                     best_start = i
    #                     best_stop = j
    # #                     best_confidence = start_scores[i] + end_scores[j]
    #                     best_confidence = math.log(start_scores[i]) + math.log(end_scores[j])
    #         return best_start, best_stop

    def find_most_confident_span(self, start_scores, end_scores):
        """ 
        Inputs: masked start_scores and end_scores of a single example
        Output: (i,j) pairs having highest Pr(i) + Pr(j)
        """
        assert len(start_scores) == len(end_scores)
        best_start = 0
        best_stop = 0
        best_confidence = -1e100
        for i in range(len(start_scores)):
            for j in range(
                min(len(end_scores), i + self.max_answer_length) - 1, i - 1, -1
            ):
                if (
                    math.log(start_scores[i]) + math.log(end_scores[j])
                    > best_confidence
                ):
                    best_start = i
                    best_stop = j
                    #                     best_confidence = start_scores[i] + end_scores[j]
                    best_confidence = math.log(start_scores[i]) + math.log(
                        end_scores[j]
                    )
        return best_start, best_stop

    def format_prediction(
        self,
        yes_no_span,
        start_scores,
        end_scores,
        sequences,
        tokens_to_text_mappings,
        official_evalutation=True,
    ):
        """
        input: all numpy arrays
        output: list of answer in english (ex: ["polar bear", "emperor penguin"])
        """
        answers = []
        assert (
            len(yes_no_span) == len(start_scores) == len(end_scores) == len(sequences)
        )

        for i in range(len(yes_no_span)):
            if official_evalutation:
                yns = yes_no_span[i].argmax(axis=-1)
                if yns == 0:
                    answers.append("yes")
                    continue
                elif yns == 1:
                    answers.append("no")
                    continue

            start, end = self.find_most_confident_span(start_scores[i], end_scores[i])

            ans = self.convert_indices_to_text(
                sequences[i], start, end, tokens_to_text_mappings[i]
            )
            answers.append(ans)

        assert len(answers) == len(sequences)

        return answers


class Evaluator:

    """ Adapted from the official HotpotQA evaluation script """

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if (
            normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            return ZERO_METRIC
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
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
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def compute_yes_no_span_em(self, gt, pred):
        pred_classes = pred.argmax(axis=-1)
        num_correct = 0
        for i in range(len(pred)):
            if pred_classes[i] == gt[i]:
                num_correct += 1
        return num_correct / len(pred)

    def evaluate_answers(
        self, gold, prediction, question_indices, divide_by_len_pred=True
    ):
        assert len(gold) >= len(prediction)
        assert len(prediction) == len(question_indices)

        metrics = {"em": 0, "f1": 0, "precision": 0, "recall": 0}

        for i in range(len(prediction)):
            em = self.exact_match_score(prediction[i], gold[question_indices[i]])
            f1, prec, recall = self.f1_score(prediction[i], gold[question_indices[i]])
            metrics["em"] += float(em)
            metrics["f1"] += f1
            metrics["precision"] += prec
            metrics["recall"] += recall

        for key in metrics.keys():
            if divide_by_len_pred:
                metrics[key] /= len(prediction)
            else:
                metrics[key] /= len(gold)

        return metrics


class QADataset(Dataset):
    def __init__(self, data, options):
        self.data = data
        assert data["max_seq_len"] == options.max_seq_len
        self.max_seq_len = options.max_seq_len
        self.max_question_len = data["max_question_len"]

    def __len__(self):
        return len(self.data["question_context_sequences"])

    def __getitem__(self, index):
        seq = torch.tensor(self.data["question_context_sequences"][index])
        segment_id = torch.tensor(self.data["segment_id"])

        assert len(self.data["answer_start_indices_offset"][index]) == len(
            self.data["answer_end_indices_offset"][index]
        )

        start_indices = torch.zeros_like(segment_id, dtype=torch.float32)
        start_indices[self.data["answer_start_indices_offset"][index]] = 1.0

        end_indices = torch.zeros_like(segment_id, dtype=torch.float32)
        end_indices[self.data["answer_end_indices_offset"][index]] = 1.0

        yes_no_span = torch.tensor(self.data["yes_no_span"][index])

        mask = torch.tensor(
            [VERY_NEGATIVE_NUMBER] * (self.max_question_len + 2)
            + [0.0] * (self.max_seq_len - self.max_question_len - 2)
        )

        if self.data["yes_no_span"][index] == 2:
            loss_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            loss_mask = torch.tensor(0.0, dtype=torch.float32)

        if len(self.data["answer_start_indices_offset"][index]) <= 0:
            first_start_index = torch.tensor(0)
            first_end_index = torch.tensor(0)
        else:
            first_start_index = torch.tensor(
                self.data["answer_start_indices_offset"][index][0]
            )
            first_end_index = torch.tensor(
                self.data["answer_end_indices_offset"][index][0]
            )

        question_index = torch.tensor(self.data["question_indices"][index])

        # return [seq, segment_id, start_indices, end_indices, yes_no_span, mask, loss_mask, question_index]
        return [
            seq,
            segment_id,
            first_start_index,
            first_end_index,
            yes_no_span,
            mask,
            loss_mask,
            question_index,
            torch.tensor(index),
        ]
        # self.data["ids_to_word_mappings"][index]]


options = options.SFOnlyOptions()
prediction_formatter = PredictionFormatter(options)
evaluator = Evaluator()

torch.cuda.set_device(options.gpu)
device = torch.device("cuda:{}".format(options.gpu))

print("Reading data pickles")

train_data = utils.unpickler(options.data_pkl_path, options.train_pkl_name)
dev_data = utils.unpickler(options.data_pkl_path, options.dev_pkl_name)

train_dataset = QADataset(train_data, options)
dev_dataset = QADataset(dev_data, options)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=options.batch_size,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=8,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
)

dev_data_loader = DataLoader(
    dev_dataset,
    batch_size=options.dev_batch_size,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=8,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
)


print("Building model")

model = QAModel(options)

print("===============================")
print("Model:")
print(model)
print("===============================")


if torch.cuda.device_count() > 1 and options.use_multiple_gpu:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

model.to(device)

criterion_yes_no_span = nn.CrossEntropyLoss()
criterion_span = nn.CrossEntropyLoss(reduction="none")
# criterion_span = nn.CrossEntropyLoss()

# Prepare optimizer
param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
# param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]


num_train_steps = int(
    (len(train_dataset) / options.batch_size / options.gradient_accumulation_steps)
    * options.epochs
)

t_total = num_train_steps
optimizer = BertAdam(
    optimizer_grouped_parameters,
    lr=options.learning_rate,
    warmup=options.warmup_proportion,
    t_total=t_total,
)


routine_log_template = "Time:{:.1f}, Epoch:{}/{}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, EM:{:.2f}, F1:{:.2f}, P:{:.2f}, R:{:.2f}, Yes_No_Span:{:.2f}"

dev_log_template = "Dev set - Exact match:{:.4f}, F1:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, Yes_No_Span:{:.2f}"

print("Training data size:{}".format(len(train_dataset)))
print("Dev data size:{}".format(len(dev_dataset)))

total_loss_since_last_time = 0
num_evaluations_since_last_best_dev_acc = 0
dev_predictions_best_model = None
stop_training_flag = False

iterations = 0
best_dev_f1 = -1
start_epoch = 0

if options.resume_training:
    if os.path.isfile(os.path.join(options.save_path, options.checkpoint_name)):
        print("=> loading checkpoint")
        checkpoint = torch.load(
            os.path.join(options.save_path, options.checkpoint_name)
        )
        start_epoch = checkpoint["epoch"]
        best_dev_f1 = checkpoint["best_acc"]
        iterations = checkpoint["iteration"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(
            "=> loaded checkpoint. Resuming epoch {}, iteration {}".format(
                checkpoint["epoch"] + 1, checkpoint["iteration"]
            )
        )

if options.dev_only:
    best_dev_f1 = -1


print("Training now")

if options.debugging_short_run:
    print("This is a short run")

start = time.time()

for epoch in range(start_epoch, options.epochs):

    for batch_idx, original_batch in enumerate(train_data_loader):

        if options.dev_only:
            break

        if iterations > num_train_steps:
            print("Reached maximum number of iterations")
            break

        if options.debugging_short_run:
            if batch_idx == options.debugging_num_iterations + 1:
                break

        batch = [t.to(device) for t in original_batch[:8]]

        model.train()
        optimizer.zero_grad()

        answer = model(sequences=batch[0], segment_id=batch[1])

        masked_start = answer[1]
        masked_end = answer[2]

        loss_span = criterion_span(masked_start, batch[2]) + criterion_span(
            masked_end, batch[3]
        )

        loss_span = (loss_span * batch[6]).mean()

        loss_yes_no_span = criterion_yes_no_span(answer[0], batch[4])

        total_loss = (
            (options.loss_weight_span * loss_span)
            + (options.loss_weight_yes_no_span * loss_yes_no_span)
        ) / (options.loss_weight_span + options.loss_weight_yes_no_span)

        # total_loss = loss_span

        total_loss_since_last_time += total_loss.item()

        total_loss.backward()

        lr_this_step = options.learning_rate * warmup_linear(
            iterations / t_total, options.warmup_proportion
        )

        if lr_this_step < 0:
            print("Learning rate < 0. Stopping training")
            break

        assert lr_this_step >= 0

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step

        optimizer.step()

        iterations += 1

        if torch.isnan(total_loss).item():
            print(
                "Loss became nan in iteration {}. Training stopped".format(iterations)
            )
            stop_training_flag = True
            break
        elif total_loss.item() < 0.0000000000001:
            print("Loss is too low. Stopping training")
            stop_training_flag = True
            break

        if iterations % options.log_every == 0:

            id_to_token_maps = []
            for index in original_batch[8]:
                mapping = train_data["ids_to_word_mappings"][index]
                id_to_token_maps.append(mapping)

            ans_str = prediction_formatter.format_prediction(
                yes_no_span=torch.softmax(answer[0], dim=-1).detach().cpu().numpy(),
                start_scores=torch.softmax(masked_start, dim=-1).detach().cpu().numpy(),
                end_scores=torch.softmax(masked_end, dim=-1).detach().cpu().numpy(),
                sequences=batch[0].detach().cpu().numpy(),
                tokens_to_text_mappings=id_to_token_maps,
                official_evalutation=False,
            )

            train_answer_metrics = evaluator.evaluate_answers(
                gold=train_data["answer_string"],
                prediction=ans_str,
                question_indices=batch[7].detach().cpu().squeeze().numpy(),
                divide_by_len_pred=True,
            )

            train_yes_no_span_em = evaluator.compute_yes_no_span_em(
                gt=batch[4].detach().cpu().numpy(),
                pred=torch.softmax(answer[0], dim=-1).detach().cpu().numpy(),
            )

            avg_loss = total_loss_since_last_time / options.log_every
            total_loss_since_last_time = 0

            print(
                routine_log_template.format(
                    time.time() - start,
                    epoch + 1,
                    options.epochs,
                    iterations,
                    avg_loss,
                    total_loss.item(),
                    train_answer_metrics["em"],
                    train_answer_metrics["f1"],
                    train_answer_metrics["precision"],
                    train_answer_metrics["recall"],
                    train_yes_no_span_em,
                )
            )

            print("Learning rate:{}".format(lr_this_step))

            if iterations % options.save_every == 0:
                snapshot_prefix = os.path.join(
                    options.save_path, options.checkpoint_name
                )
                snapshot_path = snapshot_prefix
                state = {
                    "epoch": epoch,
                    "iteration": iterations,
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_dev_f1,
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, snapshot_path)
                for f in glob.glob(snapshot_prefix + "*"):
                    if f != snapshot_path:
                        os.remove(f)

    if stop_training_flag == True:
        break

    print("Evaluating on dev set")

    # switch model to evaluation mode
    model.eval()

    dev_all_masked_start = []
    dev_all_masked_end = []
    dev_all_yes_no_span = []
    dev_question_indices = []
    with torch.no_grad():
        for dev_batch_idx, original_dev_batch in enumerate(dev_data_loader):

            if options.debugging_short_run:
                if options.debugging_num_dev_iterations == dev_batch_idx:
                    break

            dev_batch = [t.to(device) for t in original_dev_batch[:8]]
            dev_answer = model(sequences=dev_batch[0], segment_id=dev_batch[1])

            # dev_masked_start = torch.softmax(dev_answer[1] + dev_batch[5], dim=-1)
            # dev_masked_end = torch.softmax(dev_answer[2] + dev_batch[5],dim=-1)

            dev_masked_start = torch.softmax(dev_answer[1], dim=-1)
            dev_masked_end = torch.softmax(dev_answer[2], dim=-1)
            dev_yns = torch.softmax(dev_answer[0], dim=-1).detach().cpu().numpy()

            dev_question_indices.append(dev_batch[7].detach().cpu().numpy())
            dev_all_masked_start.append(dev_masked_start.detach().cpu().numpy())
            dev_all_masked_end.append(dev_masked_end.detach().cpu().numpy())
            dev_all_yes_no_span.append(dev_yns)

    dev_all_masked_start = np.concatenate(dev_all_masked_start, axis=0)
    dev_all_masked_end = np.concatenate(dev_all_masked_end, axis=0)
    dev_all_yes_no_span = np.concatenate(dev_all_yes_no_span, axis=0)
    dev_question_indices = np.concatenate(dev_question_indices, axis=0)

    dev_ans_str = prediction_formatter.format_prediction(
        yes_no_span=dev_all_yes_no_span,
        start_scores=dev_all_masked_start,
        end_scores=dev_all_masked_end,
        sequences=dev_data["question_context_sequences"],
        tokens_to_text_mappings=dev_data["ids_to_word_mappings"],
    )

    dev_answer_metrics = evaluator.evaluate_answers(
        gold=dev_data["answer_string"],
        prediction=dev_ans_str,
        question_indices=dev_question_indices,
        divide_by_len_pred=False,
    )

    dev_yes_no_span_em = evaluator.compute_yes_no_span_em(
        gt=dev_data["yes_no_span"], pred=dev_all_yes_no_span
    )

    dev_f1 = dev_answer_metrics["f1"]

    print(
        dev_log_template.format(
            dev_answer_metrics["em"],
            dev_f1,
            dev_answer_metrics["precision"],
            dev_answer_metrics["recall"],
            dev_yes_no_span_em,
        )
    )

    with open(options.acc_log_file, "a") as acc_log_file_handle:
        dev_all_metrics = dev_answer_metrics
        dev_all_metrics["yes_no_span_em"] = dev_yes_no_span_em
        acc_log_file_handle.write(
            "\n --{}-- \n Epoch :{} , metrics: {}".format(
                options.experiment_name, epoch, json.dumps(dev_all_metrics)
            )
        )

    # update best valiation set accuracy
    if dev_f1 > best_dev_f1:

        dev_predictions_best_model = [
            dev_all_yes_no_span,
            dev_all_masked_start,
            dev_all_masked_end,
        ]

        num_evaluations_since_last_best_dev_acc = 0

        # found a model with better validation set accuracy

        best_dev_f1 = dev_f1
        snapshot_prefix = os.path.join(options.save_path, "best_snapshot")
        snapshot_path = snapshot_prefix + "_dev_f1_{}_iter_{}_model.pt".format(
            dev_f1, iterations
        )

        # save model, delete previous 'best_snapshot' files
        state = {
            "epoch": epoch,
            "iteration": iterations,
            "model_state_dict": model.state_dict(),
            "best_acc": best_dev_f1,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, snapshot_path)
        for f in glob.glob(snapshot_prefix + "*"):
            if f != snapshot_path:
                os.remove(f)

        # save best predictions
        utils.pickler(
            options.save_path, options.predictions_pkl_name, dev_predictions_best_model
        )

        ans_dict = {}
        for i in range(len(dev_ans_str)):
            ans_dict[dev_data["question_ids"][i]] = dev_ans_str[i]

        # save answer strings
        utils.pickler(options.save_path, "answers_string.pkl", ans_dict)

    else:
        num_evaluations_since_last_best_dev_acc += 1

    if num_evaluations_since_last_best_dev_acc > options.early_stopping_patience:
        print(
            "Training stopped because dev acc hasn't increased in {} epochs.".format(
                options.early_stopping_patience
            )
        )
        print("Best dev set accuracy = {}".format(best_dev_f1))
        break

    if options.debugging_short_run:
        print("Short run completed")
        break

    if options.dev_only:
        break
