import os
import pandas as pd
import jieba
from gensim.models import Word2Vec
import pickle
import numpy as np


# Directory
directory = '/scratch/network/sa3937/wordembed/data-renminribao'


# Make directory for models
models = '/scratch/network/sa3937/wordembed/w2v_dec_nodownsamp/models_dec'
os.makedirs(models, exist_ok=True)


# Hyperparameters
vector_size = 300
window_size = 5
min_count_ngram = 100
epochs = 5
negative_samples = 5


# Initialize variables for decade processing
current_decade = 1940
decade_texts = []
previous_model = None


# Loop through csv files in directory
for file in sorted(os.listdir(directory)):
    year = int(file.split('.')[0])
    f = os.path.join(directory, file)
    df = pd.read_csv(f)
    print(f)
        
    # 4th column (content)
    df['content'] = df.iloc[:, 3].astype(str)
    
    # Segment content into words
    print('Segmenting...')
    df['segmented_content'] = df['content'].apply(lambda x: list(jieba.cut(x)))
    
    # Add texts to the current decade
    decade_texts.extend(df['segmented_content'])

    # Check if we need to process the decade
    if year % 10 == 9 or year == 2003:
        # Create and train Word2Vec model
        print(f'Making model for decade starting {current_decade}...')
        if previous_model is None:  # First model, initialize randomly
            model = Word2Vec(
                sentences=decade_texts, 
                vector_size=vector_size, 
                window=window_size, 
                min_count=min_count_ngram, 
                workers=4, 
                epochs=epochs, 
                negative=negative_samples
            )
        else:  # Initialize with previous decade's model
            model = Word2Vec(
                vector_size=vector_size, 
                window=window_size, 
                min_count=min_count_ngram, 
                workers=4, 
                epochs=epochs, 
                negative=negative_samples
            )
            model.build_vocab(decade_texts, update=False)

            # Align embeddings
            model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
            for word in previous_model.wv.index_to_key:
                if word in model.wv.key_to_index:
                    model.wv.vectors[model.wv.key_to_index[word]] = previous_model.wv[word]

            # Train the model
            model.train(decade_texts, total_examples=model.corpus_count, epochs=epochs)
        
        # Save Word2Vec model
        model_file_name = f'decade_{current_decade}_model.pkl'
        model_file = os.path.join(models, model_file_name)
        
        with open(model_file, 'wb') as mf:
            pickle.dump(model, mf)
        
        # Prepare for the next decade
        current_decade += 10
        decade_texts = []
        previous_model = model
        