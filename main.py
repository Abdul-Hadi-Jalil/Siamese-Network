from feature_extractor import FeatureExtractor
from siamese import SiameseNetwork
from dataset_loader import DatasetLoader
from trainer import Trainer
from pair_generator import create_balanced_pairs
import tensorflow as tf

if __name__ == "__main__":
    # Dataset path
    train_path = r"train"

    # Load dataset (returns unbatched)
    loader = DatasetLoader(train_path, target_size=(128, 128))
    dataset, class_mappings = loader.get_dataset()

    # Create pairs (now returns single pairs)
    train_pairs = create_balanced_pairs(
        dataset=dataset,
        class_mapping=class_mappings,
        samples_per_class=100
    )

    # Optional: Batch only at training time
    train_pairs = train_pairs.batch(32).prefetch(tf.data.AUTOTUNE)

    # Rest of your code remains the same...
    feature_extractor = FeatureExtractor(model_name='resnet', input_shape=(128, 128, 3))
    siamese_net = SiameseNetwork(
        feature_extractor=feature_extractor.get_model(),
        distance_metric='euclidean',
        loss_type='contrastive',
        input_shape=(128, 128, 3)
    )
    siamese_net.compile_model(optimizer='adam')

    trainer = Trainer(siamese_net)
    trainer.train(
        train_data=train_pairs,
        val_data=None,
        epochs=5,
        early_stop=False,
        checkpoint=False
    )
    trainer.save_model("siamese_resnet_bone_marrow.h5")