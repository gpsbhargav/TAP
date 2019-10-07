import os
import string
import json
import pickle


def pickler(path, pkl_name, obj):
    with open(os.path.join(path, pkl_name), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickler(path, pkl_name):
    with open(os.path.join(path, pkl_name), "rb") as f:
        obj = pickle.load(f)
    return obj


sf_pred_formatted = "path_to_predicted_supporting_facts"
ans_pred_formatted = "path_to_predicted_answers"

formatted_sf = unpickler("./", sf_pred_formatted)
ans_dict = unpickler("./", ans_pred_formatted)

final_predictions = {"answer": ans_dict, "sp": formatted_sf}

with open("predictions.json", "w") as outfile:
    json.dump(final_predictions, outfile)
