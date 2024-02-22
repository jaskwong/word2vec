import data
import onehot
import train
import numpy as np

def create_context_vector(vocab_size, word_to_id, words):
    vec = np.zeros(vocab_size)
    
    for word in words:
        vec[word_to_id[word]] += 1

    return vec


if __name__ == '__main__':
    df_data = data.read_data()
    df_processed = data.process_data(df_data)
    word_to_id, vocab_size, corpus_size = data.get_meta_data(df_processed)
    X, y = onehot.generate_training_data(df_processed, vocab_size, word_to_id, 10)
    
    # word embeddings and model
    embeddings, model = train.train(X, y, 10, 0.001, 100)

    # sample use case
    id_to_word = {v: k for k, v in word_to_id.items()}
    x = create_context_vector(vocab_size, word_to_id, ['political', 'animals', 'human', 'government'])  
    pred_x = train.predict(model, np.array([[x]]))
    pred_x_idx = np.argmax(pred_x)
    print(id_to_word[pred_x_idx])

