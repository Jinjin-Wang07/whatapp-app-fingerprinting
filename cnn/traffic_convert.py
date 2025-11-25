import sys
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import argparse
import multiprocessing as mp
from labels import labels
from sessions_plotter import session_2d_histogram, session_2d_histogram_rgb, MTU

sys.path.append(os.path.dirname(sys.path[0]))

def filter_files_by_dataset(file, dataset_list):
    """
    Check if a file matches any of the datasets in the provided list.
    
    Parameters:
    file (str): The filename to check
    dataset_list (list): List of dataset names to match against
    
    Returns:
    bool: True if file matches any dataset in the list
    """
    if not dataset_list:
        return True  # If no filter specified, include all files
    
    # Note: dataset_list is already validated in main function
    for dataset_name in dataset_list:
        # dataset_name is already cleaned, no need to strip again
        if file.find(dataset_name) >= 0:
            return True
    return False


def process_raw_dataset(bin_range, rgb, step, window_size, samples, gentest, output, input, debug=False):
    num_of_samples = 0
    dataset = []
    label = None

    # label
    # retrive the level-1 (category), -2 (app name), -3 (in-app action) labels
    (_, filename) = os.path.split(input)
    if "pdcp-encrypted_" in filename:
        labelStr = filename.split(".")[0].replace("pdcp-encrypted_", "")
    elif "pdcp-plaintext_" in filename:
        labelStr = filename.split(".")[0].replace("pdcp-plaintext_", "")
    else:
        labelStr = filename.split(".")[0]
    if labelStr in labels:
        label = labels[labelStr]
    else:
        if debug:
            print(input, flush=True)
        raise Exception("unkown label {}".format(labelStr))

    # read raw records which was parsed from pcap file.
    # each record is consisted of timestamp, direction, lcid, sequence number and length
    # df = pd.read_pickle(input, compression="gzip")
    # df = pd.read_pickle(input)
    df = pd.read_csv(input, parse_dates=['CaptureTime'])
    df["CaptureTime"] = pd.to_datetime(
        df["CaptureTime"], format="mixed", utc=True)
    df.dropna(axis=0, how="any", inplace=True)

    # drop the first 30 seconds and last 15 seconds records in case of the burst traffic when back online or re-transmisstion
    df = df[
        (df["CaptureTime"] >= df.loc[0, "CaptureTime"] + timedelta(seconds=30))
        & (
            df["CaptureTime"]
            <= df.loc[df.shape[0] - 1, "CaptureTime"] + timedelta(seconds=-15)
        )
    ].reset_index(drop=True)
    startTime = df.loc[0, "CaptureTime"]
    endTime = df.loc[df.shape[0] - 1, "CaptureTime"]

    ranges = int((endTime - startTime).total_seconds() / step)
    for sample_idx in range(ranges):
        if num_of_samples >= samples:
            break

        # fillter the time windows
        # Training sample is generated using a slice window with step size = 5 and window size = 60 in default.
        mask_start = startTime + timedelta(seconds=step) * sample_idx
        mask_end = mask_start + timedelta(seconds=window_size)
        mask = (df["CaptureTime"] >= mask_start) & (
            df["CaptureTime"] < mask_end)

        df_selected = df[mask].copy()

        df_selected["CaptureTime"] = df_selected["CaptureTime"].astype(
            np.int64) / 1e9  # get second

        # skip the windows in which less 10 records are found
        if df_selected.shape[0] < 10:
            continue
        # represent the traffic pattern whithin the time windows as PSD fingures
        if rgb:
            psd_img = session_2d_histogram_rgb(
                df_selected,
                mask_start.timestamp(),
                window_size,
                bin_range,
            )
        else:
            psd_img = session_2d_histogram(
                df_selected,
                mask_start.timestamp(),
                window_size,
                bin_range,
            )
        dataset.append(
            {"psd": psd_img, "start": mask_start, "end": mask_end, "label": label}
        )
        num_of_samples += 1
    df = pd.DataFrame(dataset)
    if gentest == False:
        # train (70%) validation (20%) test (10%)
        train, validate, test = np.split(
            df.sample(frac=1, random_state=42),
            [int(0.7 * len(df)), int(0.9 * len(df))],
        )

        # save to pickle
        train.reset_index(drop=True, inplace=True)
        train.to_pickle("{}/{}_train.pkl".format(output, labelStr))
        validate.reset_index(drop=True, inplace=True)
        validate.to_pickle("{}/{}_val.pkl".format(output, labelStr))
        test.reset_index(drop=True, inplace=True)
        test.to_pickle("{}/{}_test.pkl".format(output, labelStr))

        # output debug info
        print(">>> {} processed".format(input))
        if debug:
            print(
                ">>> {}\n|- output file {}\n|- totoal samples {}\n|- train: {}  validate:{}  test:{}".format(
                    input,
                    "{}/{}".format(output, labelStr),
                    df.shape[0],
                    train.shape[0],
                    validate.shape[0],
                    test.shape[0],
                ), flush=True
            )
    else:
        df.reset_index(drop=True, inplace=True)
        # save to pickle
        df.to_pickle("{}/{}_test.pkl".format(output, labelStr))
        # output debug info
        print(">>> {} processed".format(input))
        if debug:
            print(
                ">>> {}\n|- output file {}\n|- totoal samples {}\n".format(
                    input,
                    "{}/{}".format(output, labelStr),
                    df.shape[0],
                ), flush=True
            )


def traffic_2_psd(
    input_path,
    output_path,
    bin_range=1,
    rgb=False,
    step=5,
    window_size=60,
    samples=500,
    test=False,
    dataset_filter=None,
    num_processes=28,
    debug=False
):
    """
    Process traffic data into PSD format.

    Parameters:
    input_path (str): Path to the directory containing pickle files.
    output_path (str): Path to save the processed tf records.
    bin_range (int): Bin range for processing (default=1).
    rgb (bool): Enable uplink and downlink feature (default=False).
    step (int): Window step in seconds (default=5).
    window_size (int): Window size in seconds (default=60).
    samples (int): Maximum number of samples to process (default=500).
    test (bool): Whether to process test data (default=False).
    dataset_filter (list): List of dataset names to filter files (default=None, processes all files).
    num_processes (int): Number of parallel processes to use (default=28).
    debug (bool): Enable debug output (default=False).
    """
    if debug:
        print("Processing features of", input_path, flush=True)

    # Validate and list pickle files in the input directory
    pickle_list = os.listdir(input_path)
    pickle_list = list(filter(lambda x: x.endswith(".csv"), pickle_list))

    # Apply dataset filter if provided
    if dataset_filter:
        pickle_list = list(filter(lambda x: filter_files_by_dataset(x, dataset_filter), pickle_list))

    # Prepare arguments for parallel processing
    pickle_list = [(bin_range, rgb, step, window_size, samples, test,
                    output_path, f"{input_path}/{pickle}", debug) for pickle in pickle_list]

    # Use multiprocessing pool to process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(process_raw_dataset, pickle_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--window_size", type=int, required=True, help="Window size parameter")
    parser.add_argument("--step", type=int, required=True, help="Step parameter")
    parser.add_argument("--sample_size", type=int, default=500, help="Total sample size per file", required=False)
    parser.add_argument("--dataset_filter", type=str, default="", help="Comma-separated list of dataset names to filter (e.g., 'youtube-video,whatsapp-chat-text')", required=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    window_size = args.window_size
    step = args.step

    # Parse dataset filter if provided
    dataset_list = None
    if args.dataset_filter:
        # Clean and split dataset names
        raw_dataset_list = [name.strip() for name in args.dataset_filter.split(',') if name.strip()]
        
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
        
        # Validate dataset names
        invalid_datasets = []
        valid_dataset_list = []
        
        for dataset in raw_dataset_list:
            if dataset in valid_datasets:
                valid_dataset_list.append(dataset)
            else:
                invalid_datasets.append(dataset)
        
        # Check for invalid datasets
        if invalid_datasets:
            if args.debug:
                print("Error: Invalid dataset names detected!")
                print(f"Invalid datasets: {invalid_datasets}")
                print("\nValid dataset names are:")
                for name in sorted(valid_datasets):
                    print(f"  - {name}")
                print(f"\nValid datasets from your input: {valid_dataset_list}")
                print("Please correct the --dataset_filter parameter and try again.")
            sys.exit(1)
        
        if not valid_dataset_list:
            if args.debug:
                print("Error: No valid datasets found in the input!")
                print("Please provide valid dataset names.")
            sys.exit(1)
        
        if args.debug:
            print(f"Datasets: {valid_dataset_list}")
        dataset_list = valid_dataset_list

    # mkdir eg：{output}/w10_s5
    output_path = os.path.join(args.output, f"w{window_size}_s{step}")
    if args.debug:
        print("Output path:", output_path)
        print("window_size:", window_size, " step:", step)
    os.makedirs(output_path, exist_ok=True)

    num_processes = os.cpu_count() - 1
    sample_size = args.sample_size

    traffic_2_psd(
        input_path=args.input,
        output_path=output_path,
        bin_range=1,
        rgb=True,
        step=step,
        window_size=window_size,
        samples=sample_size,
        test=False,
        dataset_filter=dataset_list,
        num_processes=num_processes,
        debug=args.debug
    )
