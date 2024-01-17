import pandas as pd
import nltk
import numpy as np
import string
import pandas as pd
import glob
import itertools


nltk.download('punkt')
nltk.download('stopwords')

# Data files
[]

# CSV fields
ALBUM_NAME = 'album_name'
TRACK_TITLE = 'track_title'
TRACK_N = 'track_n'
LYRIC = 'lyric'
LINE = 'line'

# derived fields
TOKENS = 'tokens'
WORD_TO_ID = 'word_to_id'
ID_TO_WORD = 'id_to_word'

def read_data():
    files = glob.glob('data/*.csv')
    
    print(f'Files to read: {files}')

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df

def process_data(df):
    df_tokens = df \
        .sort_values([TRACK_TITLE, LINE], ascending=[True, True]) \
        .groupby(TRACK_TITLE) \
        .agg({LYRIC: ' '.join}) \
        .reset_index() \
        .pipe(tokenize) \
        .pipe(remove_stopwords) \
        .filter([TOKENS])
    
    print(f'Tokenized and removed stopwords for {df_tokens.shape[0]} records')
    return df_tokens

def get_meta_data(df):
    all_tokens = pd.Series(itertools.chain.from_iterable(df[TOKENS].tolist()))
    unique = all_tokens.unique()
    id_to_word = dict(enumerate(unique))
    word_to_id = {v: k for k, v in id_to_word.items()}

    vocab_size = len(word_to_id)
    corpus_size = len(all_tokens)

    print(f'Vocab size: {vocab_size} Corpus size: {corpus_size}')

    return word_to_id, vocab_size, corpus_size

def tokenize(df):
    df[TOKENS] = df[LYRIC] \
        .map(lambda l: l.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))) \
        .map(lambda l: l.lower()) \
        .map(lambda l: nltk.tokenize.word_tokenize(l))
    return df

def remove_stopwords(df):
    stopwords = nltk.corpus.stopwords.words('english')
    df[TOKENS] = df[TOKENS].map(lambda t: list(filter(lambda t: t not in stopwords, t)))
    return df