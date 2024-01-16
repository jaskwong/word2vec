import data
import pandas as pd


def get_meta_data(df):
    all_tokens = pd.Series(itertools.chain.from_iterable(df[TOKENS].tolist()))
    unique = all_tokens.unique()
    word_to_id = dict(enumerate(unique))

    vocab_size = len(word_to_id)
    corpus_size = len(all_tokens)

    print(f'Vocab size: {vocab_size} Corpus size: {corpus_size}')

    return word_to_id, vocab_size, corpus_size


if __name__ == '__main__':
    df = data.process_data()
    word_to_id, vocab_size, corpus_size = data.get_meta_data(df)