#!/bin/bash
# OriginalFullFaceSubデータに対する前処理と評価を実行するスクリプト

set -e  # エラーが発生したら停止

echo "=========================================="
echo "OriginalFullFaceSub データの前処理と評価"
echo "=========================================="

# 仮想環境のアクティベート
source venv/bin/activate

# 前処理（既に実行済みの場合はスキップ可能）
echo ""
echo "Step 1: 前処理（顔検出 + リサイズ）"
echo "----------------------------------------"
python src/preprocess_original_sub.py \
    --raw_dir raw_data/OriginalFullFaceSub \
    --output_dir data/images_original_sub \
    --labels_csv data/labels_original_sub.csv

# 評価
echo ""
echo "Step 2: モデル評価"
echo "----------------------------------------"
python src/evaluate_full.py \
    --model models/best_model.h5 \
    --labels_csv data/labels_original_sub.csv \
    --output_dir models/evaluation_original_sub

echo ""
echo "=========================================="
echo "完了！"
echo "=========================================="
echo ""
echo "評価結果は以下に保存されました："
echo "  - models/evaluation_original_sub/evaluation_results.json"
echo "  - models/evaluation_original_sub/predictions.csv"
echo "  - models/evaluation_original_sub/error_distribution.png"
echo "  - models/evaluation_original_sub/error_by_label.png"
echo ""




