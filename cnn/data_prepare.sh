# This script read the raw csv data as input, and generate data used for training and testing

# Window size and the step of sliding window
window_size=60
step=5
SAMPLE_SZ=100

## Full application list
# TRAIN_DATASET="youtube-video,tiktok-reels,vimeo,ted,WeTV,Bilibili,twitch,iQiyi,\
# spotify,youtube_music,soundcloud,qqMusic,Shazam,Kugou,NeteaseMusic,Pandora,\
# facebook-video,instagram-reels,instagram-chat-text,twitter,reddit,pinterest,Quora,Weibo,Zhihu,\
# whatsapp-chat-text,whatsapp-chat-video,messenger-chat-text,messenger-chat-video,\
# Telegram-chat-text,Telegram-chat-video,Wechat-chat-text,Wechat-chat-video,\
# Snapchat-chat-text,Snapchat-chat-video,skype-text,skype-video-call,QQ-chat-text,QQ-chat-video,\
# Line-chat-text,Line-chat-video,garena-free-file,PubgMobile,ArenaofValor,Fifa,Genshin,HearthStone,LOL,uno"

# Artifact EvaSelected application
TRAIN_DATASET="whatsapp-chat-text,whatsapp-chat-video,youtube-video,facebook-video, instagram-reels"

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSD_GEN_SCRIPT="${SCRIPT_DIR}/traffic_convert.py"
MYSCRIPT_TF_GEN="${SCRIPT_DIR}/train_toTFRecords.py"

# Set DATASET_DIR from parameter or use default
DATASET_DIR="${1:-/dataset}"
CSV_DATA_DIR="${DATASET_DIR}/raw"
PSD_OUT_DIR="${DATASET_DIR}/psd_output"
TF_OUT_DIR="${DATASET_DIR}/tfrecord"

# Feature Extraction: Parse data into PSD
echo "Converting raw traffic data into payload-size distribution (PSD) images, with window_size=${window_size}, step=${step} ..."
python -u "${PSD_GEN_SCRIPT}" --input "${CSV_DATA_DIR}" --output "${PSD_OUT_DIR}" --window_size ${window_size} --step ${step} --sample_size ${SAMPLE_SZ} --dataset_filter "${TRAIN_DATASET}" 

folder="w${window_size}_s${step}"
echo "PSD Data stored in ${PSD_OUT_DIR}/${floder}"
echo ""

# Create folders for training tfrecords and results
mkdir -p "${DATASET_DIR}/tf_output"
mkdir -p "${DATASET_DIR}/tf_output/${folder}"
mkdir -p "${DATASET_DIR}/tf_output/${folder}/labelEncoder" \
            "${DATASET_DIR}/tf_output/${folder}/model" \
            "${DATASET_DIR}/tf_output/${folder}/results" \
            "${DATASET_DIR}/tf_output/${folder}/tf-records"

echo "Loading PSD data in "${folder}" into TensorFlow's binary record file ..."
python -u ${MYSCRIPT_TF_GEN} --train_dataset "${TRAIN_DATASET}" --input "${DATASET_DIR}/psd_output/${folder}" --output "${DATASET_DIR}/tf_output/${folder}" --sample_size "${SAMPLE_SZ}"

echo "Training data stored in ${DATASET_DIR}/tf_output/${folder}/tf-records"
echo "Data preprocessing finished."
