import unittest
import pandas as pd
import onehot
import numpy as np


class TestMain(unittest.TestCase):

    word_to_id = {
        'best': 0,
        'way': 1,
        'to': 2,
        'success': 3,
        'is': 4,
        'through': 5,
        'hardwork': 6,
        'and': 7,
        'persistence': 8,
        'attitude': 9
    }

    vocab_size = 10

    def test_get_target_vectors(self):
        """Target vectors are correctly generated'"""
        
        target_vector_through = onehot.generate_target_vector(10, self.word_to_id, 'through')
        target_vector_success = onehot.generate_target_vector(10, self.word_to_id, 'success')

        self.assertListEqual(target_vector_through.tolist(), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.assertListEqual(target_vector_success.tolist(), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
    def test_get_context_vectors(self):
        """Context vectors are correctly generated'"""
        
        context_vector_through = onehot.generate_context_vector(self.vocab_size, self.word_to_id, ['to', 'through', 'to', 'success'], 'through')
        context_vector_success = onehot.generate_context_vector(self.vocab_size, self.word_to_id, ['through', 'hardwork', 'and', 'success', 'and', 'success'], 'success')

        self.assertListEqual(context_vector_through.tolist(), [0, 0, 2, 1, 0, 0, 0, 0, 0, 0])
        self.assertListEqual(context_vector_success.tolist(), [0, 0, 0, 1, 0, 1, 1, 2, 0, 0])

    def test_get_training_data(self):
        """Training data is correctly generated'"""
        
        df_tokens = pd.DataFrame([[['best', 'way', 'to', 'success', 'is', 'through', 'hardwork', 'and', 'persistence', 'and', 'attitude']]], columns=['tokens'])
        training_data = onehot.generate_training_data(df_tokens, self.vocab_size, self.word_to_id, 2)

        expected_x = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 2, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]
        
        expected_y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

        self.assertListEqual(training_data[0].tolist(), expected_x)
        self.assertListEqual(training_data[1].tolist(), expected_y)
            
