#!/bin/bash

# OriginalKomeda Training Pipeline
# 前処理 -> 学習を順次実行

set -e  # エラーが発生したら停止

echo "=========================================="
echo "OriginalKomeda Training Pipeline"
echo "=========================================="

# 設定
CONFIG_FILE="configs/komeda_default.yaml"
RAW_DATA_DIR="raw_data/OriginalKomeda"
PROCESSED_DATA_DIR="data/images_komeda"
LABELS_CSV="data/labels_komeda.csv"

# 仮想環境の有効化（既に有効化されている場合はスキップ）
if [ -d "venv" ]; then
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Virtual environment already activated."
    fi
else
    echo "Error: Virtual environment not found. Please run setup first."
    exit 1
fi

# Step 1: データ前処理
echo ""
echo "=========================================="
echo "Step 1: Data Preprocessing"
echo "=========================================="
python src/preprocess_komeda.py \
    --raw_dir "$RAW_DATA_DIR" \
    --output_dir "$PROCESSED_DATA_DIR" \
    --labels_csv "$LABELS_CSV"

# 前処理が成功したか確認
if [ ! -f "$LABELS_CSV" ]; then
    echo "Error: Preprocessing failed. Labels CSV not found."
    exit 1
fi

echo ""
echo "Preprocessing completed successfully!"

# Step 2: モデル学習
echo ""
echo "=========================================="
echo "Step 2: Model Training"
echo "=========================================="
python src/train.py --config "$CONFIG_FILE"

echo ""
echo "Training completed successfully!"

# Step 3: モデル評価
echo ""
echo "=========================================="
echo "Step 3: Model Evaluation"
echo "=========================================="

# 最新の学習済みモデルを探す
LATEST_MODEL_DIR=$(ls -td models/*/ | head -1)
BEST_MODEL="${LATEST_MODEL_DIR}best_model.h5"
CONFIG_COPY="${LATEST_MODEL_DIR}config.yaml"
EVAL_OUTPUT_DIR="${LATEST_MODEL_DIR}evaluation"

if [ -f "$BEST_MODEL" ]; then
    echo "Evaluating model: $BEST_MODEL"
    python src/evaluate.py \
        --model "$BEST_MODEL" \
        --labels_csv "$LABELS_CSV" \
        --config "$CONFIG_COPY" \
        --output_dir "$EVAL_OUTPUT_DIR"

    echo ""
    echo "Evaluation completed successfully!"
else
    echo "Error: Best model not found at $BEST_MODEL"
    exit 1
fi

# 完了メッセージ
echo ""
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo "Model directory: $LATEST_MODEL_DIR"
echo "Best model: $BEST_MODEL"
echo "Evaluation results: $EVAL_OUTPUT_DIR"
echo ""
