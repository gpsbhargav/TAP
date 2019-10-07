import os
import string
import json
import pickle
import numpy as np

import torch

import argparse


def pickler(path, pkl_name, obj):
    with open(os.path.join(path, pkl_name), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickler(path, pkl_name):
    with open(os.path.join(path, pkl_name), "rb") as f:
        obj = pickle.load(f)
    return obj


class SupportingFactFormatter:
    """
    inputs: 
    - a binary array for each question. It will have 1 if the corresponding sentence is a supporting fact 0 otherwise.
    - question id
    - names of paragraphs in the context
    - which paragraph is in which chunk
    - number of sentences in each paragraph
    
    output:
    A list like this
    [['Bridgeport, Connecticut', 5], ['Brookhaven National Laboratory', 1]]
    """

    def __init__(self, num_chunks, num_sentences_per_chunk):
        self.num_chunks = num_chunks
        self.num_sentences_per_chunk = num_sentences_per_chunk

    def find_all_indices(self, the_array, the_value):
        assert len(the_array.shape) == 1
        return list(np.where(the_array == the_value)[0])

    def find_paragraph_and_sentence_index(
        self, sent_index, paragraph_chunk_indices, num_sentences_in_paragraphs
    ):
        chunk_index = sent_index // self.num_sentences_per_chunk
        assert chunk_index < self.num_chunks
        sent_index = sent_index - (chunk_index * self.num_sentences_per_chunk)
        num_sents_cum_sum = 0
        para_index = -1
        actual_sentence_index = -1
        for p_index in paragraph_chunk_indices[chunk_index]:
            if (
                num_sents_cum_sum
                <= sent_index
                < num_sents_cum_sum + num_sentences_in_paragraphs[p_index]
            ):
                para_index = p_index
                actual_sentence_index = sent_index - num_sents_cum_sum
                break
            else:
                num_sents_cum_sum += num_sentences_in_paragraphs[p_index]
        return para_index, actual_sentence_index

    def find_paragraph_name(self, para_index, paragraph_names):
        assert 0 <= para_index
        return paragraph_names[para_index]

    def format_supporting_facts(
        self,
        predictions,
        question_ids,
        paragraph_names,
        paragraph_chunk_indices,
        num_sentences_in_paragraphs,
    ):
        assert (
            len(predictions)
            == len(question_ids)
            == len(paragraph_names)
            == len(paragraph_chunk_indices)
            == len(num_sentences_in_paragraphs)
        )

        out_records = {}
        for i, pred_row in enumerate(predictions):
            indices_of_sf = self.find_all_indices(the_array=pred_row, the_value=1)
            formatted_sf_list = []
            for sf_idx in indices_of_sf:
                para_idx, sentence_idx = self.find_paragraph_and_sentence_index(
                    sent_index=sf_idx,
                    paragraph_chunk_indices=paragraph_chunk_indices[i],
                    num_sentences_in_paragraphs=num_sentences_in_paragraphs[i],
                )
                if para_idx < 0 or sentence_idx < 0:
                    continue
                para_name = self.find_paragraph_name(
                    para_index=para_idx, paragraph_names=paragraph_names[i]
                )
                formatted_sf_list.append([para_name, int(sentence_idx)])
            out_records[question_ids[i]] = formatted_sf_list

        return out_records


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--threshold", type=float, required=True)
command_line_args = parser.parse_args()


threshold = command_line_args.threshold

dataset_path = "./"
dataset_pkl_name = "bm_preprocessed_dev.pkl"

sf_formatter = SupportingFactFormatter(num_chunks=4, num_sentences_per_chunk=18)

dev_data = unpickler(dataset_path, dataset_pkl_name)
pred_raw = unpickler("./", "bm_predictions.pkl")

dev_all_sf = pred_raw[3]
dev_thresholded_sf = (torch.tensor(dev_all_sf) > threshold).numpy()

dev_sf_formatted = sf_formatter.format_supporting_facts(
    predictions=dev_thresholded_sf,
    question_ids=dev_data["question_ids"],
    paragraph_names=dev_data["paragraph_names"],
    paragraph_chunk_indices=dev_data["paragraph_chunk_indices"],
    num_sentences_in_paragraphs=dev_data["num_sentences_in_paragraphs"],
)

pickler("./", command_line_args.name, dev_sf_formatted)

