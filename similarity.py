import os
import numpy as np
from gensim.models import Word2Vec
import pickle
import matplotlib.pyplot as plt


# 3 words
word1 = '消防队员'
word2 = '女人'  # woman
word3 = '男人'  # man
    
output_file = 'similarity.txt'

# Load the saved Word2Vec model
models = '/scratch/network/sa3937/wordembed/w2v_dec_nopunc/models_dec'
with open(output_file, 'w') as f_out:
    for file in os.listdir(models):
        if file.endswith('_model.pkl'):
            model_file = os.path.join(models, file)
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        
            # Save similarity results to file
            f_out.write(f'Loaded model from {file}\n')
            
            similarity12 = similarity13 = similarity23 = None
            
            # Similarity between 1st and 2nd words
            if word1 in model.wv and word2 in model.wv:
                similarity12 = model.wv.similarity(word1, word2)
                f_out.write(f'Similarity between "{word1}" and "{word2}": {similarity12}\n')
            else:
                f_out.write(f'One or both words "{word1}" or "{word2}" are not in vocabulary\n')
                
            # Similarity between 1st and 3rd words
            if word1 in model.wv and word3 in model.wv:
                similarity13 = model.wv.similarity(word1, word3)
                f_out.write(f'Similarity between "{word1}" and "{word3}": {similarity13}\n')
            else:
                f_out.write(f'One or both words "{word1}" or "{word3}" are not in vocabulary\n')
                
            # Similarity between 2nd and 3rd words
            if word2 in model.wv and word3 in model.wv:
                similarity23 = model.wv.similarity(word2, word3)
                f_out.write(f'Similarity between "{word2}" and "{word3}": {similarity23}\n')
            else:
                f_out.write(f'One or both words "{word2}" or "{word3}" are not in vocabulary\n')
                