MODEL_URL="https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4.zip"
MODEL_DIR="./app/analysis/models"
MODEL_NAME="metrabs_eff2s_y4"
ZIP_FILE="$MODEL_NAME.zip"

if [ -d "$MODEL_DIR/$MODEL_NAME" ]; then
  echo "$MODEL_DIR/$MODEL_NAME already exists, skipping download"
else
  echo "Downloading $MODEL_NAME from $MODEL_URL"

  if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory $MODEL_DIR does not exist"
    exit 1
  fi

  curl -L "$MODEL_URL" -o "$MODEL_DIR/$ZIP_FILE"

  echo "Unzipping"
  unzip -q "$MODEL_DIR/$ZIP_FILE" -d "$MODEL_DIR"

  echo "Cleaning up"
  rm "$MODEL_DIR/$ZIP_FILE"

  echo "$MODEL_NAME downloaded and extracted to $MODEL_DIR"
fi

pyinstaller serve_mac.py \
  --console \
  --onedir  \
  --distpath ./pyinstaller_builds \
  --add-data "VideoAnalysisToolBackend:VideoAnalysisToolBackend" \
  --add-data "app:app" \
  --clean \
  --noconfirm \
  --collect-all numpy \
  --collect-all scipy \
  --collect-all tensorflow \
  --collect-all keras \
  --collect-all torch \
  --collect-all torchvision \
  --collect-all opencv-python \
  --collect-all pandas \
  --collect-all matplotlib \
  --hidden-import=scipy._lib.array_api_compat.numpy.fft