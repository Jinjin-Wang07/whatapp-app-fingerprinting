import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import utils
import argparse
import multiprocessing as mp

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("No GPU detected — using CPU only.")
else:
    print(f"Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# print("TensorFlow version:", tf.__version__)

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

def load_pickle(filename, flag, config):
    if not config["train_label_encoder"]:
        print(">>> reading {} with flag {}".format(filename, flag))
    
    df = pd.read_pickle(filename)
    df["label_1"] = df["label"].apply(lambda x: "{}".format(x[0]))  # first category
    df["label_2"] = df["label"].apply(lambda x: "{}".format(x[1]))  # second category
    df["label_3"] = df["label"].apply(
        lambda x: "{}-{}".format(x[1], x[2])
    )  # third category
    
    # Calculate sample sizes based on 7:2:1 ratio
    sample_size = config.get("sample_size", 500)
    train_size = int(sample_size * 0.7)  # 70%
    val_size = int(sample_size * 0.2)    # 20%
    test_size = int(sample_size * 0.1)   # 10%
    
    # The samples have already been split into train, val, and test in traffic_convert.py
    # Here, we only load a certain amount of data from each respective PSD pickle file.
    if flag == "train":
        df = df.sample(n=min(train_size, len(df)), random_state=1, replace=True)
    elif flag == "val":
        df = df.sample(n=min(val_size, len(df)), random_state=1, replace=True)
    elif flag == "test":
        df = df.sample(n=min(test_size, len(df)), random_state=1, replace=True)
    
    if config["train_label_encoder"]:
        df = df[["label_1", "label_2", "label_3"]]
    if config["for_apps_unseen"]:
        if flag == "test":
            df = df[df["label_2"].isin(config["apps_unseen"])]
        else:
            df = df[~df["label_2"].isin(config["apps_unseen"])]
    return df


def load_dataset(filenames, flag, config):
    # print("loading {} dataset".format(flag))
    filenames = [(filename, flag, config) for filename in filenames]
    with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
        df = pd.concat(pool.starmap(load_pickle, filenames), copy=False).reset_index(
            drop=True
        )
    # print(df.shape)
    return df


def to_tfrecords(dataset, output, flag, idx):
    with tf.io.TFRecordWriter(
        "{}/tf-records/{}_{}.tfrecord".format(
            output,
            flag,
            idx,
        )
    ) as writer:
        for _, data in dataset.iterrows():
            feature = {
                "start": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[bytes("{}".format(data["start"]), encoding="utf8")]
                    )
                ),
                "end": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[bytes("{}".format(data["end"]), encoding="utf8")]
                    )
                ),
                "psd": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=data["psd"].reshape(-1))
                ),
                "psd_shape": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=data["psd"].shape)
                ),
                "label_1": tf.train.Feature(
                    float_list=tf.train.FloatList(value=data["label_1"])
                ),
                "label_2": tf.train.Feature(
                    float_list=tf.train.FloatList(value=data["label_2"])
                ),
                "label_3": tf.train.Feature(
                    float_list=tf.train.FloatList(value=data["label_3"])
                ),
            }
            # print(f'Array length == {data["label_1"]} {data["label_2"]} {data["label_3"]}') 
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    pass


def build_dataset(flag, input, output, train_dataset, config):
    labels = train_dataset
    labels = list(filter(lambda x: len(x) > 0, labels))
    
    # load dataset
    if flag == "train" or config["train_label_encoder"] == True:
        dataset = load_dataset(
            ["{}/{}_train.pkl".format(input, filename) for filename in labels], "train", config
        )
    elif flag == "val":
        dataset = load_dataset(
            ["{}/{}_val.pkl".format(input, filename) for filename in labels],
            "validation", config
        )
    elif flag == "test":
        dataset = load_dataset(
            ["{}/{}_test.pkl".format(input, filename) for filename in labels], "test", config
        )
    
    # train label encoder
    if config["train_label_encoder"] == True:
        # print("train_label_encoding")
        for key in ["label_1", "label_2", "label_3"]:
            num_of_classes = len(dataset[key].unique())
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(dataset[key])
            path1 = "{}/labelEncoder/classes_{}_{}.npy".format(
                output,
                key,
                fileFlag(
                    config["for_apps_unseen"], config["for_apps_idle"], config["for_pdcp_eth"], config["for_top5"], config["for_grey"]
                ),
            )
            path2 = "{}/labelEncoder/num_of_classes_{}_{}.npy".format(
                output,
                key,
                fileFlag(
                    config["for_apps_unseen"], config["for_apps_idle"], config["for_pdcp_eth"], config["for_top5"], config["for_grey"]
                ),
            )
            # print("Lable: {}".format(label_encoder.classes_))
            # print(
            #     "saving {} classes {}\npath: {}\npath: {}".format(
            #         key, num_of_classes, path1, path2
            #     )
            # )
            np.save(path1, label_encoder.classes_)
            np.save(path2, num_of_classes)

        return
    # encode label
    for key in ["label_1", "label_2", "label_3"]:
        num_of_classes = np.load(
            "{}/labelEncoder/num_of_classes_{}_{}.npy".format(
                output,
                key,
                fileFlag(
                    config["for_apps_unseen"], config["for_apps_idle"], config["for_pdcp_eth"], config["for_top5"], config["for_grey"]
                ),
            ),
            allow_pickle=True,
        )
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(
            "{}/labelEncoder/classes_{}_{}.npy".format(
                output,
                key,
                fileFlag(
                    config["for_apps_unseen"], config["for_apps_idle"], config["for_pdcp_eth"], config["for_top5"], config["for_grey"]
                ),
            ),
            allow_pickle=True,
        )
        # print(
        #     "{} num_of_classes: {}\nLabel: {}".format(
        #         key, num_of_classes, label_encoder.classes_
        #     )
        # )

        # preprare tf-records
        if config["for_apps_unseen"] == True and (key == "label_2" or key == "label_3"):
            continue
        dataset[key] = pd.Series(
            list(
                utils.to_categorical(
                    label_encoder.transform(dataset[key]), num_of_classes
                )
            )
        )
    # save dataset as tf-records
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    data = np.array_split(dataset, 20)
    with mp.Pool(processes=20) as pool:    
        args = [(t, output, flag, idx) for idx, t in enumerate(data)]
        pool.starmap(to_tfrecords, args)

def train2TFRecords(input_path,
    output_path,
    train_dataset,
    flag="train",
    apps_unseen=False,
    apps_idle=False,
    pdcp_eth=False,
    top5=False,
    train_label_encoder=False,
    grey=False,
    sample_size=500
    ):

    config = {
        "num_of_classes": 0,
        "apps_unseen": ["6", "7", "14", "15", "22", "23", "30", "31", "38", "39"],
        "for_apps_unseen": apps_unseen,
        "for_apps_idle": apps_idle,
        "for_pdcp_eth": pdcp_eth,
        "for_top5": top5,
        "for_grey": grey,
        "train_label_encoder": train_label_encoder,
        "input_path": input_path,
        "output_path": output_path,
        "train_dataset": train_dataset,
        "flag": flag,
        "sample_size": sample_size
    }

    build_dataset(
        flag=config["flag"],
        input=config["input_path"],
        output=config["output_path"],
        train_dataset=config["train_dataset"],
        config=config
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="Input path", required=True)
    parser.add_argument("--output", type=str, default="", help="Output path", required=True)
    parser.add_argument("--train_dataset", type=str, default="", help="File list", required=True)
    parser.add_argument("--flag", type=str, help="train, val, test", required=False, default="train")
    parser.add_argument("--apps_unseen", type=bool, default=False, help="For apps-unseen", required=False)
    parser.add_argument("--apps_idle", type=bool, default=False, help="For apps-idle", required=False)
    parser.add_argument("--pdcp_eth", type=bool, default=False, help="For pdcp_eth", required=False)
    parser.add_argument("--top5", type=bool, default=False, help="For top5", required=False)
    parser.add_argument("--train_label_encoder", type=bool, default=False, help="Train LabelEncoder", required=False)
    parser.add_argument("--grey", type=bool, default=False, help="Process grey encode psd", required=False)
    parser.add_argument("--sample_size", type=int, default=500, help="Total sample size per file", required=False)

    args = parser.parse_args()
    train_list = [name.strip() for name in args.train_dataset.split(',') if name.strip()]

    # Define valid dataset names
    valid_datasets = {
        "youtube-video", "tiktok-reels", "vimeo", "ted", "WeTV", "Bilibili", "twitch", "iQiyi",
        "spotify", "youtube_music", "soundcloud", "qqMusic", "Shazam", "Kugou", "NeteaseMusic", "Pandora",
        "facebook-video", "instagram-reels", "instagram-chat-text", "twitter", "reddit", "pinterest", "Quora", "Weibo", "Zhihu",
        "whatsapp-chat-text", "whatsapp-chat-video", "messenger-chat-text", "messenger-chat-video",
        "Telegram-chat-text", "Telegram-chat-video", "Wechat-chat-text", "Wechat-chat-video",
        "Snapchat-chat-text", "Snapchat-chat-video", "skype-text", "skype-video-call", "QQ-chat-text", "QQ-chat-video",
        "Line-chat-text", "Line-chat-video", "garena-free-file", "PubgMobile", "ArenaofValor", "Fifa", "Genshin", "HearthStone", "LOL", "uno"
    }
    
    # Validate train_list
    invalid_datasets = []
    valid_train_list = []
    
    for dataset in train_list:
        if dataset in valid_datasets:
            valid_train_list.append(dataset)
        else:
            invalid_datasets.append(dataset)
    
    # Check for invalid datasets
    if invalid_datasets:
        print("Error: Invalid dataset names detected!")
        print(f"Invalid datasets: {invalid_datasets}")
        print("\nValid dataset names are:")
        for name in sorted(valid_datasets):
            print(f"  - {name}")
        print(f"\nValid datasets from your input: {valid_train_list}")
        print("Please correct the --train_dataset parameter and try again.")
        sys.exit(1)
    
    if not valid_train_list:
        print("Error: No valid datasets found in the input!")
        print("Please provide valid dataset names.")
        sys.exit(1)
    
    print(f"Using data: {valid_train_list}")
    train_list = valid_train_list

    sample_size = args.sample_size

    # Get LabelEncoder
    train2TFRecords(
        input_path=args.input,
        output_path=args.output,
        train_dataset=train_list,
        flag="train",
        apps_unseen=args.apps_unseen,
        apps_idle=args.apps_idle,
        pdcp_eth=args.pdcp_eth,
        top5=args.top5,
        train_label_encoder=True,
        grey=args.grey,
        sample_size=sample_size,
    )

    # Training data
    train2TFRecords(
        input_path=args.input,
        output_path=args.output,
        train_dataset=train_list,
        flag="train",
        apps_unseen=args.apps_unseen,
        apps_idle=args.apps_idle,
        pdcp_eth=args.pdcp_eth,
        top5=args.top5,
        train_label_encoder=args.train_label_encoder,
        grey=args.grey,
        sample_size=sample_size,
    )

    # Validation data
    train2TFRecords(
        input_path=args.input,
        output_path=args.output,
        train_dataset=train_list,
        flag="val",
        apps_unseen=args.apps_unseen,
        apps_idle=args.apps_idle,
        pdcp_eth=args.pdcp_eth,
        top5=args.top5,
        train_label_encoder=args.train_label_encoder,
        grey=args.grey,
        sample_size=sample_size,
    )

    # Test data
    train2TFRecords(
        input_path=args.input,
        output_path=args.output,
        train_dataset=train_list,
        flag="test",
        apps_unseen=args.apps_unseen,
        apps_idle=args.apps_idle,
        pdcp_eth=args.pdcp_eth,
        top5=args.top5,
        train_label_encoder=args.train_label_encoder,
        grey=args.grey,
        sample_size=sample_size,
    )