import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2 # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore

class FeatureExtractor:
    def __init__(self, model_name="cnn", input_shape=(128, 128, 3), freeze=True):
        """
        Initializes the feature extractor.

        Args:
            model_name (str): The type of feature extractor ('cnn', 'resnet', 'inception', 'mobilenet').
            input_shape (tuple): The shape of input images.
        """
        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.freeze = freeze
        self.model = self._build_model()

    def _build_model(self):
        """Builds the selected feature extractor model."""
        if self.model_name == "resnet":
            return self._resnet()
        elif self.model_name == "inception":
            return self._inception()
        elif self.model_name == "mobilenet":
            return self._mobilenet()
        else:
            return self._cnn()  # Default CNN model

    def _cnn(self):
        """Simple custom CNN feature extractor."""
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        model = Model(inputs, x, name="Custom_CNN")
        return model

    def _resnet(self):
        """ResNet50 as feature extractor (pretrained on ImageNet)."""
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape)
        if self.freeze:
            base_model.trainable = False
        x = Flatten()(base_model.output)
        model = Model(base_model.input, x, name="ResNet50_FeatureExtractor")
        return model

    def _inception(self):
        """InceptionV3 as feature extractor (pretrained on ImageNet)."""
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=self.input_shape)
        x = Flatten()(base_model.output)
        model = Model(base_model.input, x, name="InceptionV3_FeatureExtractor")
        return model

    def _mobilenet(self):
        """MobileNetV2 as feature extractor (pretrained on ImageNet)."""
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=self.input_shape)
        x = Flatten()(base_model.output)
        model = Model(base_model.input, x, name="MobileNetV2_FeatureExtractor")
        return model

    def get_model(self):
        """Returns the built feature extractor model."""
        return self.model
