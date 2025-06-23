import os
import tensorflow as tf

class Trainer:
    """Handles training, evaluation, and saving/loading of a Siamese Network."""

    def __init__(self, siamese_network, save_dir="checkpoints", log_dir="logs"):
        """
        Initializes the trainer.

        Parameters:
        - siamese_network: Instance of the trained SiameseNetwork class.
        - save_dir (str): Directory to save trained model weights.
        - log_dir (str): Directory for TensorBoard logs.
        """
        self.model = siamese_network.get_model()
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Ensure save directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def train(self, train_data, val_data=None, epochs=10, early_stop=True, checkpoint=True):
        callbacks = []
        
        if early_stop and val_data:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True))

        if checkpoint:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.save_dir, "siamese_best.h5"),
                monitor="val_loss" if val_data else "loss",
                save_best_only=True,
                verbose=1
            ))

        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.log_dir))

        # Modified fit() - removed batch_size
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        return history
 

    def evaluate(self, test_data):
        """
        Evaluates the trained model.

        Parameters:
        - test_data: Test dataset.

        Returns:
        - Loss and accuracy of the model on test set.
        """
        results = self.model.evaluate(test_data)
        print(f"Test Loss: {results}")
        return results

    def save_model(self, filename="siamese_final.h5"):
        """
        Saves the trained model.

        Parameters:
        - filename (str): Name of the file to save the model.
        """
        model_path = os.path.join(self.save_dir, filename)
        self.model.save(model_path)
        print(f"Model saved at: {model_path}")

    def load_model(self, filepath):
        """
        Loads a trained model.

        Parameters:
        - filepath (str): Path to the saved model.
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")