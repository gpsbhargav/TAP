# TAP
Training and evaluation of "Translucent Answer Prediction" on HotpotQA

## Requirements
- Python >= 3.6
- Pytorch >= 1.0
- Numpy = 1.15.4
- scikit-learn = 0.20.1
- usjon = 1.35
- jupyterlab = 0.35.3
- NLTK = 3.4
- PyTorch implementation of BERT from https://github.com/gpsbhargav/pytorch-pretrained-BERT

## Instructions for training TAP
- Download training and development sets of HotpotQA distractor setting into TAP/hotpotqa/. Google's pretrained model and vocab are also inside a docker image on dockerhub (docker pull gpsbhargav/iljddtrywo). The same image also contains the trained TAP and was submitted for official evaluation and leaderboard entry.
- Preprocess the data for the bottom machine by runing preproc_1.ipynb and preproc_2.ipynb (twice each. With the "TRAINING" flag set to True and False)in TAP/bottom_machine/data/ans_and_sf/.
- Create a directory called results in TAP/bottom_machine/
- Training bottom machine: run TAP/bottom_machine/code/train_sf_only.py. This will also generate predictions on the dev set in .TAP/bottom_machine/results/<experiment_name>.
- Run TAP/sf_formatter.py. Give the predictions of the bottom machine (on the dev set) to this file as an input
- Copy the output of the previous step into TAP/top_machine/data/sf_only/
- Preprocess the data for the top machine by running TAP/top_machine/data/sf_only/preprocessing.ipynb (With the "TRAINING" flag set to True and False). Remember to set the path to the output of the previous step in this notebook. The supporting facts predicted by the bottom machine will be used to create the dev set for the top machine.
- Training top machine: First create a directory TAP/top_machine/results/. Then run TAP/top_machine/code/train.py. This step will output the predictions of the dev set as well in TAP/top_machine/results/<experiment_name>. 
- Paste in the paths to the formatted predictions of the bottom machine (output of TAP/sf_formatter.py) and the top machine in the appropriate places in TAP/merge_both_predictions.py and run it. This will output predictions.json. 
- Post-process the predicted answers by running answer_post_processing.sh. This will fix the issues caused by the tokenization and output pred.json. These are the final predictions on the dev set and can be given to the official evaluation script (given on HotpotQA's website)


The file TAP/bottom_machine/code/train_ans_and_sf.py trains a model that predicts both answer and supporting facts.  

