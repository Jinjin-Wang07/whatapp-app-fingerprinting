import pandas as pd

"""labels
fist class: video:0, music:1, social:2, communication:3, game:4, others: 5
second class: app name
third class: activity type
"""
labels = {
    # video
    "youtube-video": [0, 0, 0],
    "tiktok-reels": [0, 1, 0],
    "vimeo": [0, 2, 0],
    "ted": [0, 3, 0],
    "WeTV": [0, 4, 0],
    "Bilibili": [0, 5, 0],
    "twitch": [0, 6, 0],
    "iQiyi": [0, 7, 0],
    # music
    "spotify": [1, 8, 0],
    "youtube_music": [1, 9, 0],
    "soundcloud": [1, 10, 0],
    "qqMusic": [1, 11, 0],
    "Shazam": [1, 12, 0],
    "Kugou": [1, 13, 0],
    "NeteaseMusic": [1, 14, 0],
    "Pandora": [1, 15, 0],
    # social
    "facebook-video": [2, 16, 0],
    "instagram-reels": [2, 17, 0],
    "instagram-chat-text": [2, 17, 1],
    "twitter": [2, 18, 0],
    "reddit": [2, 19, 0],
    "pinterest": [2, 20, 0],
    "Quora": [2, 21, 0],
    "Weibo": [2, 22, 0],
    "Zhihu": [2, 23, 0],
    # communication
    "whatsapp-chat-text": [3, 24, 0],
    "whatsapp-chat-video": [3, 24, 1],
    "messenger-chat-text": [3, 25, 0],
    "messenger-chat-video": [3, 25, 1],
    "Telegram-chat-text": [3, 26, 0],
    "Telegram-chat-video": [3, 26, 1],
    "Wechat-chat-text": [3, 27, 0],
    "Wechat-chat-video": [3, 27, 1],
    "Snapchat-chat-text": [3, 28, 0],
    "Snapchat-chat-video": [3, 28, 1],
    "skype-text": [3, 29, 0],
    "skype-video-call": [3, 29, 1],
    "QQ-chat-text": [3, 30, 0],
    "QQ-chat-video": [3, 30, 1],
    "Line-chat-text": [3, 31, 0],
    "Line-chat-video": [3, 31, 1],
    # game
    "garena-free-file": [4, 32, 0],
    "PubgMobile": [4, 33, 0],
    "ArenaofValor": [4, 34, 0],
    "Fifa": [4, 35, 0],
    "Genshin": [4, 36, 0],
    "HearthStone": [4, 37, 0],
    "LOL": [4, 38, 0],
    "uno": [4, 39, 0],
    # others
    "chrome": [5, 40, 0],
    "google-photos": [5, 41, 0],
    "idle": [5, 42, 0],
    # validation
    "top5": [99, 99, 99],
}


def labels_df():
    df = pd.DataFrame(
        labels,
    ).T
    df["label_2"] = df.apply(lambda row: "{}".format(row[1]), axis=1)
    df["label_3"] = df.apply(lambda row: "{}-{}".format(row[1], row[2]), axis=1)
    return df
