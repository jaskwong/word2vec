import unittest
import pandas as pd
import data


class TestData(unittest.TestCase):
    df_data = pd.DataFrame(
        columns=['album_name', 'track_title', 'track_n', 'lyric', 'line'])
    df_processed = pd.DataFrame(columns=['tokens'])

    def tearDown(self) -> None:
        self.df_data.drop(self.df_data.index, inplace=True)
        self.df_processed.drop(self.df_processed.index, inplace=True)

    def test_processed_columns(self):
        """Processed dataframe contains a single 'tokens' column"""

        self.add_song('album name', 'track title', 1, "test lyric", 1)

        df_processed = data.process_data(self.df_data)
        cols = df_processed.columns

        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0], 'tokens')

    def test_tokenized_lines(self):
        """Processed dataframe 'tokens' column contains array of tokens split on whitespace'"""

        self.add_song('album name', 'track title', 1, "test lyric", 1)

        df_processed = data.process_data(self.df_data)
        tokens = df_processed['tokens'].iloc[0]

        self.assertEqual(tokens, ['test', 'lyric'])

    def test_get_meta_data(self):
        """Meta data is correctly retrieved from a tokens dataframe'"""

        self.add_processed(['a', 'bend', 'in', 'the', 'road', 'is'])
        self.add_processed(['not', 'the', 'end', 'of', 'the', 'road'])

        word_to_id, vocab_size, corpus_size = data.get_meta_data(self.df_processed)

        self.assertEqual(vocab_size, 9)
        self.assertEqual(corpus_size, 12)

        self.assertEqual(len(word_to_id), 9)
        self.assertIn('a', word_to_id)
        self.assertIn('bend', word_to_id)
        self.assertIn('in', word_to_id)
        self.assertIn('the', word_to_id)
        self.assertIn('road', word_to_id)
        self.assertIn('is', word_to_id)
        self.assertIn('not', word_to_id)
        self.assertIn('end', word_to_id)
        self.assertIn('of', word_to_id)

    def add_song(self, album_name, track_title, track_n, lyric, line):
        self.df_data.loc[len(self.df_data.index)] = \
            [album_name, track_title, track_n, lyric, line]
        
    def add_processed(self, tokens):
        self.df_processed.loc[len(self.df_processed.index)] = [tokens]
