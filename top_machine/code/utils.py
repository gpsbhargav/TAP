from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np
import nltk
import random
import math
import os
import pickle
from tqdm import tqdm

import torch

import pdb


def pickler(path, pkl_name, obj):
    with open(os.path.join(path, pkl_name), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickler(path, pkl_name):
    with open(os.path.join(path, pkl_name), "rb") as f:
        obj = pickle.load(f)
    return obj
