import numpy as np
import os
os.chdir("understanding_new_words")
embedding = np.load("embedding.npy")
print(embedding[0])
from word2vec_basic import *

print dictionary[:10]