import data
import onehot

if __name__ == '__main__':
    df_data = data.read_data()
    df_processed = data.process_data(df_data)
    word_to_id, vocab_size, corpus_size = data.get_meta_data(df_processed)
    training_data = onehot.generate_training_data(df_processed, vocab_size, word_to_id, 9)