import random
import tensorflow as tf

def create_balanced_pairs(dataset, class_mapping, samples_per_class=100):
    # Create reverse mapping (index to class name)
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    # Convert dataset to list since we'll sample randomly
    dataset_list = list(dataset.as_numpy_iterator())
    
    # Group images by class index
    class_images = {class_idx: [] for class_idx in class_mapping.values()}
    for img, lbl in dataset_list:
        class_idx = lbl.item()  # Convert numpy to Python int
        class_images[class_idx].append(img)
    
    pairs = []
    labels = []
    class_indices = list(class_mapping.values())

    # Create positive pairs (same class)
    for class_idx in class_indices:
        images = class_images[class_idx]
        if len(images) < 2:
            continue
            
        for _ in range(samples_per_class):
            img1, img2 = random.sample(images, 2)
            pairs.append((img1, img2))
            labels.append(1)  # 1 = similar

    # Create negative pairs (different classes)
    for _ in range(samples_per_class * len(class_indices)):
        class1, class2 = random.sample(class_indices, 2)
        img1 = random.choice(class_images[class1])
        img2 = random.choice(class_images[class2])
        pairs.append((img1, img2))
        labels.append(0)  # 0 = dissimilar

    # Create dataset without batching
    def gen():
        for (img1, img2), label in zip(pairs, labels):
            yield (img1, img2), label
            
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(10000)