import tensorflow as tf

class LossFunction:
    """A unified class to handle different loss functions for a Siamese Network."""

    def __init__(self, loss_type="contrastive", margin=1.0, alpha=0.2):
        """
        Initialize the loss function.

        Parameters:
        - loss_type (str): The type of loss function ('contrastive', 'triplet', 'multilet').
        - margin (float): The margin value used in contrastive and triplet loss.
        - alpha (float): The scaling factor used in certain loss functions.
        """
        self.loss_type = loss_type.lower()
        self.margin = margin
        self.alpha = alpha

    def compute_loss(self, y_true, y_pred):
        """Computes the selected loss function dynamically."""
        if self.loss_type == "contrastive":
            return self._contrastive_loss(y_true, y_pred)
        elif self.loss_type == "triplet":
            return self._triplet_loss(y_true, y_pred)
        elif self.loss_type == "multilet":
            return self._multilet_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_type}")

    def _contrastive_loss(self, y_true, y_pred):
        """Computes Contrastive Loss."""
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

    def _triplet_loss(self, y_true, y_pred):
        """Computes Triplet Loss (y_pred contains [anchor, positive, negative])."""
        anchor, positive, negative = y_pred
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def _multilet_loss(self, y_true, y_pred):
        """Computes Multilet Loss using Huber loss."""
        return tf.reduce_mean(tf.losses.huber(y_true, y_pred))

    def __call__(self, y_true, y_pred):
        """Allows the class instance to be used as a function."""
        return self.compute_loss(y_true, y_pred)
    
    def get_config(self):
        return {
            "loss_type": self.loss_type,
            "margin": self.margin,
            "alpha": self.alpha
        }