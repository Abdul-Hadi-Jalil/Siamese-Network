import tensorflow as tf
import os

class DatasetLoader:
    def __init__(self, dataset_path, target_size=(128, 128), batch_size=32, has_subfolders=True, shuffle=True):
        """
        Initializes the dataset loader.

        Args:
            dataset_path (str): Path to the dataset folder.
            target_size (tuple): Target size for resizing images.
            batch_size (int): Number of images per batch.
            has_subfolders (bool): Whether dataset has subfolders for classes.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.has_subfolders = has_subfolders
        self.shuffle = shuffle

        self.dataset, self.class_mapping = self._load_dataset()

    def _process_image(self, file_path, label):
        """
        Reads and processes an image.

        Args:
            file_path (str): Path to the image file.
            label (int): Numeric label for the image.

        Returns:
            (tf.Tensor, tf.Tensor): Processed image and its label.
        """
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.target_size) / 255.0  # Normalize
        return image, label

    def _load_dataset(self):
        """
        Loads dataset images and assigns numeric labels automatically.

        Returns:
            dataset (tf.data.Dataset): TensorFlow dataset.
            class_mapping (dict): Mapping from class names to numeric labels.
        """
        image_paths = []
        labels = []

        if self.has_subfolders:
            # Load dataset with subfolders as class labels
            class_names = sorted(os.listdir(self.dataset_path))
            class_mapping = {name: idx for idx, name in enumerate(class_names)}

            for class_name in class_names:
                class_folder = os.path.join(self.dataset_path, class_name)
                if os.path.isdir(class_folder):
                    for img_file in os.listdir(class_folder):
                        image_paths.append(os.path.join(class_folder, img_file))
                        labels.append(class_mapping[class_name])
        else:
            # Load dataset with no subfolders (single class)
            image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
            labels = [0] * len(image_paths)  # Assign a single label

            class_mapping = {"all": 0}

        # Convert lists to TensorFlow tensors
        image_paths_tensor = tf.constant(image_paths)
        labels_tensor = tf.constant(labels, dtype=tf.int32)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, labels_tensor))
        dataset = dataset.map(self._process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(1000)

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset, class_mapping

    def get_dataset(self):
        """
        Returns the dataset.

        Returns:
            (tf.data.Dataset, dict): The TensorFlow dataset and class mapping.
        """
        return self.dataset.unbatch(), self.class_mapping