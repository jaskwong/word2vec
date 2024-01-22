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
    print(X)
    model = train.train(X, y, 100, 0.001, 100) 

    id_to_word = inv_map = {v: k for k, v in word_to_id.items()}

    test = create_context_vector(vocab_size, word_to_id, ["sensible", "incredible", "friends", "jealous", "respects"])  
    pred = train.predict(model, np.array([[test]]))
    pred_idx = np.argmax(pred)
    print(id_to_word[pred_idx])

