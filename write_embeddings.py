import gensim
import os

# Make directory for word embeddings
embed = '/scratch/network/sa3937/wordembed/w2v_dec_nopunc/word_embeddings'
os.makedirs(embed, exist_ok=True)

current_decade = 1940

# Path to the models directory
models = '/scratch/network/sa3937/wordembed/w2v_dec_nopunc/models_dec'

# Iterate through the model files and process them
for file in os.listdir(models):
    if file.endswith('_model.pkl'):
        model_file = os.path.join(models, file)
        
        with open(model_file, 'rb') as f:
            model = gensim.models.Word2Vec.load(model_file)
            
        # Create a new file for the current decade
        embed_file_name = f'word_embeddings_{current_decade}.txt'
        embed_file = os.path.join(embed, embed_file_name)
        
        # Write the embeddings to the file
        with open(embed_file, 'w') as f_out:
            for word in model.wv.index_to_key:
                vector = model.wv[word]
                vector_str = ' '.join(map(str, vector))
                f_out.write(f"{word} {vector_str}\n")
        
        current_decade += 10

print("Done")