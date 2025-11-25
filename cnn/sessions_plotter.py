import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

MTU = 1500


def session_2d_histogram(df, startTime, windowsSize, binRange=1):
    # ts = df[df["Direction"] == 1]["CaptureTime"].tolist()
    # sizes = df[df["Direction"] == 1]["Length"].tolist()
    ts = df["CaptureTime"].tolist()
    sizes = df["Length"].tolist()
    # normalize the timestamp
    # ts_norm = list(map(int, ((np.array(ts) - ts[0]) / max_delta_time) * MTU))
    ts_norm = ((np.array(ts) - startTime) / windowsSize) * MTU
    H, xedges, yedges = np.histogram2d(
        sizes,
        ts_norm,
        bins=(range(0, MTU + 1, binRange), range(0, MTU + 1, binRange)),
    )

    return H.astype(np.uint16)


def session_2d_histogram_rgb(df, startTime, windowsSize, binRange=1):
    # # directions for uplink and downlink
    mask_uplink = df["Direction"] == 1
    mask_downlink = ~mask_uplink  # df["Direction"] == 0

    uplink_data = df.loc[mask_uplink, ["CaptureTime", "Length"]].to_numpy(dtype=np.float64)
    downlink_data = df.loc[mask_downlink, ["CaptureTime", "Length"]].to_numpy(dtype=np.float64)

    ts_uplink, sizes_uplink = uplink_data[:, 0], uplink_data[:, 1]
    ts_downlink, sizes_downlink = downlink_data[:, 0], downlink_data[:, 1]

    # normalize the timestamp
    ts_uplink_norm = ((np.array(ts_uplink) - startTime) / windowsSize) * MTU
    ts_downlink_norm = ((np.array(ts_downlink) - startTime) / windowsSize) * MTU
    
    bins = range(0, MTU + 1, binRange)

    def fast_histogram2d(x, y, bins):
        return np.histogram2d(x, y, bins=[bins, bins])

    H_uplink, xedges_uplink, yedges_uplink = np.histogram2d(sizes_uplink, ts_uplink_norm, bins)
    H_downlink, xedges_downlink, yedges_downlink = np.histogram2d(sizes_downlink, ts_downlink_norm, bins)

    # # Red
    # H_uplink = H_uplink.reshape(H_uplink.shape[0], H_uplink.shape[1], 1)

    # # Green
    # H_downlink = H_downlink.reshape(H_downlink.shape[0], H_downlink.shape[1], 1)
    # # combine the uplink and downlink
    # H = np.dstack((H_uplink, H_downlink))

    H = np.stack((H_uplink, H_downlink), axis=-1)

    return H.astype(np.uint16)
