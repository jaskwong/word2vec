import data
import numpy as np
import pandas as pd

def generate_training_data(df, vocab_size, word_to_id, window_size):
    x = []
    y = []

    for tokens in df["tokens"]:
        for idx, token in enumerate(tokens):
            target_vector = generate_target_vector(vocab_size, word_to_id, token)
            y.append(target_vector)

            window_start = max(0, idx - window_size)
            window_end = min(idx + window_size + 1, len(tokens))
            context_vector = generate_context_vector(vocab_size, word_to_id, tokens[window_start:window_end], token)
            x.append(context_vector)
            
    return np.stack(np.array([x, y]))

def generate_target_vector(vocab_size, word_to_id, word):
    vec = np.zeros(vocab_size)
    vec[word_to_id[word]] = 1
    return vec

def generate_context_vector(vocab_size, word_to_id, context_words, target_word):
    vec = np.zeros(vocab_size)
    
    for word in context_words:
        vec[word_to_id[word]] += + 1

    vec[word_to_id[target_word]] -= 1
    return vec


if __name__ == '__main__':
    df_data = data.read_data()
    df_processed = data.process_data(df_data)
    word_to_id, vocab_size, corpus_size = data.get_meta_data(df_processed)
    generate_training_data(df_processed.iloc[0:2], vocab_size, word_to_id, 9)