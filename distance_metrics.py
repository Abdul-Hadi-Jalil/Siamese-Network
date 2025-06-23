import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore

class DistanceMetrics:
    @staticmethod
    def euclidean(embedding1, embedding2):
        """Computes Euclidean distance between two embeddings."""
        return tf.sqrt(tf.reduce_sum(tf.square(embedding1 - embedding2), axis=-1, keepdims=True))

    @staticmethod
    def manhattan(embedding1, embedding2):
        """Computes Manhattan (L1) distance between two embeddings."""
        return tf.reduce_sum(tf.abs(embedding1 - embedding2), axis=-1, keepdims=True)

    @staticmethod
    def absolute_difference(embedding1, embedding2):
        """Computes the absolute difference between two embeddings."""
        return tf.abs(embedding1 - embedding2)

    @staticmethod
    def ann(input_dim):
        """Creates an ANN-based similarity metric."""
        model = Sequential([
            Input(shape=(input_dim * 2,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Output similarity score (0 to 1)
        ])
        return model
