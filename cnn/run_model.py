import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from functools import partial
from model import cnn_model
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from metrics import precision, recall, f1_score, top_2_categorical_accuracy
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sklearn_f1_score, classification_report

# print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("No GPU detected — using CPU only.")
else:
    print(f"Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

def fileFlag(forUnseen, forIdle, forPDCPeth, forTop5, forGrey):
    flag = "apps"
    if forUnseen:
        flag += "-unseen"
    if forIdle:
        flag += "-idle"
    if forPDCPeth:
        flag += "-pdcpEth"
    if forTop5:
        flag += "-top5"
    if forGrey:
        flag += "-grey"
    if flag == "apps":
        flag += "-all"
    return flag

def load_num_classes_from_file(dataset_path, key, config=None):
    """Read the number of classes from the file"""
    if config is None:
        flag = fileFlag(False, False, False, False, False)
    else:
        flag = fileFlag(
            config.get("for_apps_unseen", False),
            config.get("for_apps_idle", False), 
            config.get("for_pdcp_eth", False),
            config.get("for_top5", False),
            config.get("for_grey", False)
        )
    
    file_path = "{}/labelEncoder/num_of_classes_{}_{}.npy".format(
        dataset_path, key, flag
    )
    
    try:
        num_classes = int(np.load(file_path, allow_pickle=True))
        return num_classes
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}, using default size")
        if key == "label_1":
            return 5
        elif key == "label_2":
            return 40
        elif key == "label_3":
            return 49
        else:
            return 10

def read_tfrecord(example, config, labeled=True, time=False):
    
    dataset_path = config.get("dataset_path", "")
    label_1_size = load_num_classes_from_file(dataset_path, "label_1", config)
    label_2_size = load_num_classes_from_file(dataset_path, "label_2", config)
    label_3_size = load_num_classes_from_file(dataset_path, "label_3", config)
    
    feature_description = {
        "start": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "end": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "psd": tf.io.FixedLenFeature(shape=config["IMAGE_SIZE"], dtype=tf.int64),
        "psd_shape": tf.io.FixedLenFeature([3], dtype=tf.int64),
        "label_1": tf.io.FixedLenFeature([label_1_size], dtype=tf.float32),
        "label_2": tf.io.FixedLenFeature([label_2_size], dtype=tf.float32),
        "label_3": tf.io.FixedLenFeature([label_3_size], dtype=tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    if time:
        return (
            example["psd"],
            example["label_{}".format(config["label_class"])],
            example["start"],
            example["end"],
        )
    return example["psd"], example["label_{}".format(config["label_class"])]


def load_dataset(filenames, config, labeled=True, time=False):
    options = tf.data.Options()
    # ignore_order.experimental_deterministic = False  # disable order, increase speed
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    raw_dataset = tf.data.TFRecordDataset(filenames)
    raw_dataset = raw_dataset.with_options(options)
    parsed_dataset = raw_dataset.map(
        partial(read_tfrecord, config=config, labeled=labeled, time=time), num_parallel_calls=32
    )
    return parsed_dataset


def get_dataset(filenames, config, labeled=True, time=False):
    # print(filenames)
    dataset = load_dataset(filenames, config, labeled=labeled, time=time)
    if config["train"]:
        dataset = dataset.shuffle(
            buffer_size=1000, reshuffle_each_iteration=True if config["train"] else False
        )
    dataset = dataset.batch(config["BATCH_SIZE"]).prefetch(10)
    return dataset

def run_model(train, path, label_class, model_name, epochs=300, apps_unseen=False, apps_idle=False, pdcp_eth=False, top5=False, grey=False):
    config = {
        "idle": False,
        "unseen": True,
        "model_name": None,
        "AUTOTUNE": tf.data.AUTOTUNE,
        "BATCH_SIZE": 32,
        "IMAGE_SIZE": [1500, 1500, 2],
        "NUM_CLASSES": 0,
        "train": False,
        "label_class": 0,
        "LOG_DIR": os.path.join(path, "results"),
        "tf_callback": None,
        "dataset_path": path,
        "for_apps_unseen": apps_unseen,
        "for_apps_idle": apps_idle,
        "for_pdcp_eth": pdcp_eth,
        "for_top5": top5,
        "for_grey": grey,
    }
    bin = 1

    config["train"] = train
    config["model_name"] = model_name
    config["label_class"] = label_class
    config["IMAGE_SIZE"] = [int(1500 / bin), int(1500 / bin), 2]
    config["tf_callback"] = tf.keras.callbacks.TensorBoard(log_dir=config["LOG_DIR"])
    
    def load(prefix, count, time=False):
        return get_dataset(
            ["{}/tf-records/{}_{}.tfrecord".format(path, prefix, i) for i in range(count)],
            config = config,
            time=time
        )
    
    if config["train"]:
        train_dataset = load("train", 20)
        val_dataset = load("val", 20)
    else:
        test_dataset = load("test", 20)
        test_dataset_time = load("test", 20, time=True)

    if config["train"] == True:
        for images, labels in train_dataset.take(
            1
        ):  # only take first element of dataset
            config["NUM_CLASSES"] = labels.shape[1]
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = cnn_model(input_shape=config["IMAGE_SIZE"], num_of_classes=config["NUM_CLASSES"])
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            validation_batch_size=4,
            callbacks=[config["tf_callback"]],
            verbose=1,
        )
        model.save(
            "{}/model/{}".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            )
        )
        print(
            "Saving model to {}/model/{}".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            )
        )
    else:
        print(
            "Loading model from {}/model/{}".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            )
        )
        labels_true = np.concatenate(
            [y for x, y, start, end in test_dataset_time], axis=0
        )
        np.save(
            "{}/results/{}_true.npy".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            ),
            labels_true,
        )
        np.save(
            "{}/results/{}_true_start.npy".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            ),
            np.concatenate([start for x, y, start, end in test_dataset_time], axis=0),
        )
        np.save(
            "{}/results/{}_true_end.npy".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            ),
            np.concatenate([end for x, y, start, end in test_dataset_time], axis=0),
        )
        model = keras.models.load_model(
            "{}/model/{}".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            ),
            custom_objects={
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "top_2_categorical_accuracy": top_2_categorical_accuracy,
            },
        )
        predictions = model.predict(test_dataset, verbose=1)
        np.save(
            "{}/results/{}_pred.npy".format(
                path, "{}_label{}_bin{}".format(model_name, label_class, bin)
            ),
            predictions,
        )
        
        # Print prediction results and evaluation metrics
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS:")
        print("=" * 80)
        
        # Convert predictions and true labels to class indices
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels_indices = np.argmax(labels_true, axis=1)  # Convert from one-hot encoding
        
        # Print basic information
        print(f"Number of test samples: {len(true_labels_indices)}")
        print(f"Number of classes: {predictions.shape[1]}")
        print("-" * 80)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels_indices, predicted_labels)
        precision_score_val = precision_score(true_labels_indices, predicted_labels, average='weighted', zero_division=0)
        recall_score_val = recall_score(true_labels_indices, predicted_labels, average='weighted', zero_division=0)
        f1_score_val = sklearn_f1_score(true_labels_indices, predicted_labels, average='weighted', zero_division=0)
        
        # Print performance metrics
        print("PERFORMANCE METRICS:")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision_score_val:.4f}")
        print(f"Recall:    {recall_score_val:.4f}")
        print(f"F1-Score:  {f1_score_val:.4f}")
        print("-" * 80)
        
        # Print detailed classification report
        print("-" * 80)
        print("DETAILED CLASSIFICATION REPORT:")
        report = classification_report(true_labels_indices, predicted_labels, zero_division=0)
        print(report)


def str2bool(v):
    """Convert string to boolean value for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str2bool, default=False, help="train or test (True/False)")
    parser.add_argument("--input", type=str, default="", help="dataset path", required=True)
    parser.add_argument("--label", type=int, default=3, help="label class", required=False)
    parser.add_argument("--model_name", type=str, default="", help="model name", required=True)
    parser.add_argument("--epochs", type=int, default=300, help="number of training epochs", required=False)
    parser.add_argument("--apps_unseen", type=str2bool, default=False, help="For apps-unseen (True/False)", required=False)
    parser.add_argument("--apps_idle", type=str2bool, default=False, help="For apps-idle (True/False)", required=False)
    parser.add_argument("--pdcp_eth", type=str2bool, default=False, help="For pdcp_eth (True/False)", required=False)
    parser.add_argument("--top5", type=str2bool, default=False, help="For top5 (True/False)", required=False)
    parser.add_argument("--grey", type=str2bool, default=False, help="Process grey encode psd (True/False)", required=False)
    args = parser.parse_args()

    run_model(
        train=args.train,
        path=args.input,
        label_class=args.label,
        model_name=args.model_name,
        epochs=args.epochs,
        apps_unseen=args.apps_unseen,
        apps_idle=args.apps_idle,
        pdcp_eth=args.pdcp_eth,
        top5=args.top5,
        grey=args.grey
    )