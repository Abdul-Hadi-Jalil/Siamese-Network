from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Lambda # type: ignore
import tensorflow as tf
from distance_metrics import DistanceMetrics
from loss import LossFunction

class SiameseNetwork:
    def __init__(self, feature_extractor, distance_metric="euclidean", loss_type="contrastive", input_shape=(128, 128, 3)):
        """
        Initializes the Siamese Network.
        
        Parameters:
        - feature_extractor: The feature extraction model
        - distance_metric (str): Distance metric ('euclidean', 'manhattan', 'absolute_difference')
        - loss_type (str): Loss function ('contrastive', 'triplet', 'multilet')
        - input_shape (tuple): Shape of the input images
        """
        self.feature_extractor = feature_extractor
        self.distance_metric = distance_metric.lower()
        self.loss_type = loss_type
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Builds the complete Siamese network architecture."""
        # Input layers for two images
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        
        # Generate features using shared weights
        processed_a = self.feature_extractor(input_a)
        processed_b = self.feature_extractor(input_b)
        
        # Distance computation using DistanceMetrics class
        def get_distance(embeddings):
            if self.distance_metric == "euclidean":
                return DistanceMetrics.euclidean(embeddings[0], embeddings[1])
            elif self.distance_metric == "manhattan":
                return DistanceMetrics.manhattan(embeddings[0], embeddings[1])
            elif self.distance_metric == "absolute_difference":
                return DistanceMetrics.absolute_difference(embeddings[0], embeddings[1])
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        distance = Lambda(get_distance)([processed_a, processed_b])
        
        # Final model
        return Model(inputs=[input_a, input_b], outputs=distance)

    def compile_model(self, optimizer="adam"):
        """Compiles the model with the selected loss function."""
        loss = LossFunction(loss_type=self.loss_type)
        self.model.compile(optimizer=optimizer, loss=loss)

    def get_model(self):
        """ Returns the compiled siamese model """
        return self.model