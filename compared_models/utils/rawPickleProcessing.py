import os
from datetime import timedelta
import pandas as pd
from multiprocessing import Process, Manager, Lock
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_features(input_df):
    """
        Extract features as in Identifie what are you doing paper from dataframe
    """
    df_uplink = input_df[input_df["Direction"] == 1]
    df_downlink = input_df[input_df["Direction"] == 0]
    def extract(df):
        features = {}
        features["num"] = df.shape[0]
        # packet size
        features["vol"] = df["Length"].sum()
        features["max_s"] = df["Length"].max()
        features["min_s"] = df["Length"].min()
        features["mean_s"] = df["Length"].mean()
        features["std_s"] = df["Length"].std()
        features["20_s"] = df["Length"].quantile(0.2)
        features["40_s"] = df["Length"].quantile(0.4)
        features["60_s"] = df["Length"].quantile(0.6)
        features["80_s"] = df["Length"].quantile(0.8)
        
        # IAT
        df1 = df.copy()
        df1["IAT"] = (df['CaptureTime'] - df['CaptureTime'].shift(1)).dt.total_seconds()  ## First item is NAN
        # df["IAT"] = df["IAT"].round(3)
        features["max_iat"] = df1["IAT"].max()
        features["min_iat"] = df1["IAT"].min()
        features["mean_iat"] = df1["IAT"].mean()
        features["std_iat"] = df1["IAT"].std()
        # Percentile
        features["20_iat"] = df1["IAT"].quantile(0.2)  # 20th Percentile
        features["40_iat"] = df1["IAT"].quantile(0.4)  # 40th Percentile
        features["60_iat"] = df1["IAT"].quantile(0.6)  # 60th Percentile
        features["80_iat"] = df1["IAT"].quantile(0.8)  # 80th Percentile
        return features
    up_featrues = extract(df_uplink)
    up_featrues = {"up_" + str(key): val for key, val in up_featrues.items()}
    down_features = extract(df_downlink)
    down_features = {"down_" + str(key): val for key, val in down_features.items()}
    return up_featrues|down_features # Set Union


# Read the input dataframe
# Return features accroding to sample_duration and window_size
# sample_duration: duration of sample will be processed
# window_size: One-time Window Size
# independent_window: Samples are independent for the model
def process_file(df, sample_duration, window_size, sliding_window=False, sliding_wnd_step = 5,drop_empty=True, start_time = None, max_samples = 2000):
    dataset = []

    startTime = df.loc[0, "CaptureTime"]

    if start_time:
        startTime += start_time
    
    # endTime = df.loc[df.shape[0] - 1, "CaptureTime"]
    endTime = startTime + timedelta(seconds=sample_duration)

    if endTime > df.loc[df.shape[0] - 1, "CaptureTime"]:
        endTime = df.loc[df.shape[0] - 1, "CaptureTime"]

    num_sample = 0

    if not sliding_window:
        ranges = int((endTime - startTime).total_seconds() / window_size)
    
        # print(ranges)
        for i in range(ranges):
            mask_start = startTime + timedelta(seconds=window_size) * i
            mask_end = startTime +  (i+1)*timedelta(seconds=window_size)
            mask = (df["CaptureTime"] >= mask_start) & (
                df["CaptureTime"] <= mask_end)
            df_mask = df[mask].reset_index(drop=True)
            # print(df_mask.shape[0])
            if df_mask.shape[0] > 1 and drop_empty==True:
                features = extract_features(df_mask)
                dataset.append(features)
                num_sample += 1

            if num_sample >= max_samples:
                break
        return dataset

    # Sliding window
    else:
        step = sliding_wnd_step
        ranges = int((endTime - startTime).total_seconds() / step)
    
        for i in range(ranges):
            mask_start = startTime + timedelta(seconds=step) * i
            mask_end = startTime + \
                timedelta(seconds=step) * i + timedelta(seconds=window_size)

            # When reach the end
            if mask_end > endTime:
                break
                
            mask = (df["CaptureTime"] >= mask_start) & (
                df["CaptureTime"] <= mask_end)
            df_mask = df[mask].reset_index(drop=True)
            if df_mask.shape[0] > 5:
                features = extract_features(df_mask)
                dataset.append(features)
                num_sample += 1

            if num_sample >= max_samples:
                break
        return dataset


def load_valid_pkl_files(input_folder):
    """
    Read .pkl files with suffix _XXhXXm.pkl from the input folder, the suffix represent the record time duration
    Load the file which has more than 10mins record
    
    Args:
    input_folder (str): the path of the folder containing the .pkl files.
    
    Returns:
    list: a list of loaded DataFrames.
    """
    
    loaded_files = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pkl"):
            base_name, suffix = os.path.splitext(file_name)
            time_str = base_name.split('_')[-1]
            # print(time_str)
            if "h" in time_str and "m" in time_str:
                hours = int(time_str.split('h')[0])
                minutes = int(time_str.split('h')[1].replace('m', ''))

                # Calculate total minutes, and ignore files which time < 10mins
                total_minutes = hours * 60 + minutes
                if total_minutes < 10:
                    # print(f"Skipping file {file_name} (Duration: {total_minutes} minutes)")
                    continue

                # load .pkl
                file_path = os.path.join(input_folder, file_name)
                df = pd.read_pickle(file_path)
                loaded_files.append((file_name, df))
                print(f"Loaded file: {file_name} (Duration: {total_minutes} minutes)")
            else:
                print(f"Skipping file {file_name} (Invalid time format in name)")
    return loaded_files

def load_pkl_files(input_folder, file_name_list, cps='infer'):
    loaded_files = []
    for file_name in file_name_list:
        file_path = input_folder + "/" + file_name
        df = pd.read_pickle(file_path, compression=cps)
        loaded_files.append((file_name, df))
        print(f"Loaded file: {file_name}")
    return loaded_files

def load_all_pkl_files(input_folder, cps='infer', drop_head=False):
    """
    Read all .pkl files from the input folder
    
    Args:
    input_folder (str): the path of the folder containing the .pkl files.
    
    Returns:
    list: a list of loaded DataFrames.
    """
    
    loaded_files = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pkl"):
            # Load the file
            file_path = os.path.join(input_folder, file_name)
            # print(f"Loaded file: {file_path}")
            df = pd.read_pickle(file_path, compression=cps)
            if drop_head:
                start = df["CaptureTime"].iloc[0]
                delta = timedelta(seconds=180)
                first_idx = (df["CaptureTime"] >= start + delta).idxmax()
                df = df.loc[first_idx:].reset_index(drop=True)
            loaded_files.append((file_name, df))
            print(f"Loaded file: {file_name}")
    return loaded_files

def _load_single_csv(input_folder, file_name, raw_excel=False):
    file_path = os.path.join(input_folder, file_name)
    if not raw_excel:
        df = pd.read_csv(file_path, parse_dates=['CaptureTime'])
        df["CaptureTime"] = pd.to_datetime(df["CaptureTime"], format="mixed", utc=True)
        df.dropna(axis=0, how="any", inplace=True)
        # print(f"Loaded file: {file_name}")
        return file_name, df
    else:
        colnames = ["CaptureTime", "RNTI", "Direction", "LCID", "SequenceNumber", "Length"]
        colDtypes = {
            "CaptureTime": object,
            "RNTI": "Int64",
            "Direction": object,
            "LCID": "Int64",
            "SequenceNumber": "Int64",
            "Length": "Int64",
        }
        df = pd.read_csv(file_path, names=colnames, header=None, delimiter=";", dtype=colDtypes)
        df["CaptureTime"] = pd.to_datetime(df["CaptureTime"], errors="coerce", utc=True)
        df["Direction"] = df["Direction"].map({"Uplink": 0, "Downlink": 1})
        df.dropna(axis=0, how="any", inplace=True)
        # print(f"Loaded file: {file_name}")
        return file_name, df

def load_all_csv_files(input_folder,  raw_excel=False):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    loaded_files = []
    # print(f"Loading csv data...")
    with ProcessPoolExecutor(max_workers=27) as executor:
        futures = {executor.submit(_load_single_csv, input_folder, file_name, raw_excel): file_name for file_name in csv_files}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                result = future.result()
                loaded_files.append(result)
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
    return loaded_files
    
def extract_pdcp_features_from_dfs(L, sample_duration, window_size=60, startTime=180, step=5, max_samples=2000, sliding_window=True):
    """
    Eextract features from raw pdcp dataframes

    Args:
    L: List of tuples with format (file_name, dataframe)

    Returns:
    df_features_dict: Dict
    """
    def worker(file_name, df, shared_dict, lock, st, step, max_samples):
        # print(file_name)
        dataset = process_file(df, sample_duration, window_size, start_time=st, sliding_window=sliding_window, sliding_wnd_step = step, max_samples=max_samples)
        with lock:
            shared_dict[file_name] = pd.DataFrame(dataset).fillna(0)
            # shared_dict[file_name]["IAT"] = shared_dict[file_name]["IAT"].fillna(0)
        # df_features_dict[label].to_pickle("{}/prepared_{}.pkl".format(path, label))
    
    # Extract features in parallel
    df_features_dict = {}
    keys = []
    startTime = timedelta(seconds=startTime) # remove first 3 mins
    
    with Manager() as manager:
        shared_dict = manager.dict()
        lock = Lock()
        
        processes = []
        for file_name, df in L:
            keys.append(file_name)
            p = Process(target=worker, args=(file_name, df, shared_dict, lock, startTime, step, max_samples))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        df_features_dict = dict(shared_dict)
        print("All files processed")
        
    return keys, df_features_dict
