import os
import numpy as np
from gensim.models import Word2Vec
import pickle

word1 = '女人'  # woman
word2 = '男人'  # man
word3 = '妈妈'  # mom
word4 = '爸爸'  # dad
    
output_file = 'analogy.txt'

# Load the saved Word2Vec model
models = '/scratch/network/sa3937/wordembed/w2v_dec_nodownsamp/models_dec'
with open(output_file, 'w') as f_out:
    for file in os.listdir(models):
        if file.endswith('_model.pkl'):
            model_file = os.path.join(models, file)
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        
            # Save checks to file
            f_out.write(f'Loaded model from {file}\n')
            
            if word1 in model.wv and word2 in model.wv and word3 in model.wv and word4 in model.wv:
                closest = model.wv.most_similar(positive=[word1, word4], negative=[word2], topn=10)
                f_out.write(f'The closest words to "woman + dad - man" are "{closest}"\n')
                similarity = model.wv.similarity(word3, closest[0][0])
                f_out.write(f'Similarity between "mom" and analogy vector: {similarity}\n')