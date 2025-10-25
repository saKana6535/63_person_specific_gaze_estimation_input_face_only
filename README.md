# FullFace Gaze Estimation - 顔画像のみを用いた個人特化型視線推定

論文「Person-Specific Gaze Estimation from Low-Quality Webcam Images」の実装です。

## 概要

このプロジェクトは、1人の顔画像のみを入力として、スクリーン座標(x, y)を推定する個人特化型の視線推定モデルを構築します。

## 環境構築

### 1. Python環境のセットアップ

Python 3.10を使用します。pyenvを使ってバージョンを設定：

```bash
pyenv local 3.10.13
```

### 2. 仮想環境の作成とライブラリのインストール

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## データセット

### FullFace（学習用）

データセットは既に `raw_data/FullFace/` ディレクトリに配置されています。

- 形式: `face_{x}_{y}_{id}.jpg`
- 例: `face_1645_1064_3.jpg` → x=1645, y=1064
- サンプル数: 約11,800枚

### OrginalFullFace（評価用）

`raw_data/OrginalFullFace/` ディレクトリに配置されています。

- ディレクトリ構造：ラベル名ごとにディレクトリ分け
  - `topLeft/`: 画面左上 (20, 20)
  - `topRight/`: 画面右上 (1780, 20)
  - `bottomLeft/`: 画面左下 (20, 1149)
  - `bottomRight/`: 画面右下 (1780, 1149)
  - `center/`: 画面中央 (880, 564.5)
- サンプル数: 各ラベル20枚、合計100枚
- 前処理: Haar Cascadeによる顔検出が必要

### OriginalFullFaceSub（評価用）

`raw_data/OriginalFullFaceSub/` ディレクトリに配置されています。

- ファイル名から座標ラベルを抽出
  - `topLeft.png`: 画面左上 (20, 20)
  - `topRight.png`: 画面右上 (1780, 20)
  - `bottomLeft.png`: 画面左下 (20, 1149)
  - `bottomRight.png`: 画面右下 (1780, 1149)
  - `center.png`: 画面中央 (880, 564.5)
- サンプル数: 各ラベル1枚、合計5枚
- 前処理: Haar Cascadeによる顔検出が必要

## 使用方法

### 簡単な実行（推奨）

全パイプライン（前処理 → 学習 → 評価）を一括実行：

```bash
./run.sh
```

### OrginalFullFaceデータでの評価

学習済みモデル (`models/best_model.h5`) を使って、OrginalFullFaceデータを評価：

```bash
./evaluate_original.sh
```

このスクリプトは以下を実行します：
1. Haar Cascadeによる顔検出と224×224へのリサイズ
2. 学習済みモデルでの評価（全100枚）

評価結果は `models/evaluation_original/` に保存されます。

### OriginalFullFaceSubデータでの評価

学習済みモデル (`models/best_model.h5`) を使って、OriginalFullFaceSubデータを評価：

```bash
./evaluate_original_sub.sh
```

このスクリプトは以下を実行します：
1. Haar Cascadeによる顔検出と224×224へのリサイズ
2. 学習済みモデルでの評価（全5枚）

評価結果は `models/evaluation_original_sub/` に保存されます。

### 個別実行

#### 1. データ前処理（FullFace）

```bash
python src/preprocess.py \
    --raw_dir raw_data/FullFace \
    --output_dir data/images \
    --labels_csv data/labels.csv
```

#### 2. データ前処理（OrginalFullFace）

```bash
python src/preprocess_original.py \
    --raw_dir raw_data/OrginalFullFace \
    --output_dir data/images_original \
    --labels_csv data/labels_original.csv
```

#### 2-2. データ前処理（OriginalFullFaceSub）

```bash
python src/preprocess_original_sub.py \
    --raw_dir raw_data/OriginalFullFaceSub \
    --output_dir data/images_original_sub \
    --labels_csv data/labels_original_sub.csv
```

#### 3. モデル学習

```bash
python src/train.py --config configs/ff_default.yaml
```

#### 4. モデル評価（train/test分割あり）

```bash
python src/evaluate.py \
    --model models/{timestamp}/best_model.h5 \
    --labels_csv data/labels.csv \
    --config models/{timestamp}/config.yaml \
    --output_dir models/{timestamp}/evaluation
```

#### 5. モデル評価（全データ使用）

```bash
python src/evaluate_full.py \
    --model models/best_model.h5 \
    --labels_csv data/labels_original.csv \
    --output_dir models/evaluation_original
```

## モデル構成

```
Conv2D (32 filters, 7x7, ReLU)
Conv2D (96 filters, 7x7, ReLU)
MaxPooling2D (2x2)
BatchNormalization
Conv2D (160 filters, 7x7, ReLU)
BatchNormalization
Flatten
Dense (64 units, ReLU)
Dropout (0.1)
Dense (2 units, Linear)  # 出力: (x, y)座標
```

## ハイパーパラメータ

デフォルト設定（`configs/ff_default.yaml`）：

- **Learning Rate**: 1e-4
- **Kernel Size**: 7
- **Dropout Rate**: 0.1
- **Epochs**: 100（Early Stopping併用）
- **Batch Size**: 32
- **Train/Test Split**: 80/20

## 評価指標

- **MAE (Mean Absolute Error)**: ピクセル単位の平均絶対誤差
- **Angular Error**: 角度誤差（度）
  - 計算式: `atan(error_px / ppi × 2.54 / distance_cm)`
  - デフォルト: PPI=96, Distance=60cm

## 出力

学習後、以下のファイルが `models/{timestamp}/` に保存されます：

- `best_model.h5`: 最良モデル（val_maeが最小）
- `final_model.h5`: 最終エポックのモデル
- `config.yaml`: 使用した設定
- `training_log.csv`: 学習履歴（エポック毎の損失・メトリクス）
- `history.json`: 学習履歴のJSON形式
- `test_metrics.json`: テストセットでの評価結果
- `tensorboard/`: TensorBoardログ
- `evaluation/`: 評価結果
  - `evaluation_results.json`: 評価メトリクス
  - `predictions.csv`: 全予測結果
  - `error_distribution.png`: 誤差分布のグラフ

## TensorBoardでの学習監視

```bash
tensorboard --logdir models/{timestamp}/tensorboard
```

## プロジェクト構成

```
.
├── raw_data/
│   ├── FullFace/                  # 元データセット（学習用）
│   ├── OrginalFullFace/           # 評価用データセット（100枚）
│   │   ├── topLeft/
│   │   ├── topRight/
│   │   ├── bottomLeft/
│   │   ├── bottomRight/
│   │   └── center/
│   └── OriginalFullFaceSub/       # 評価用データセット（5枚）
│       ├── topLeft.png
│       ├── topRight.png
│       ├── bottomLeft.png
│       ├── bottomRight.png
│       └── center.png
├── data/
│   ├── images/                    # 前処理後の画像（FullFace）
│   ├── labels.csv                 # ラベル情報（FullFace）
│   ├── images_original/           # 前処理後の画像（OrginalFullFace）
│   ├── labels_original.csv        # ラベル情報（OrginalFullFace）
│   ├── images_original_sub/       # 前処理後の画像（OriginalFullFaceSub）
│   └── labels_original_sub.csv    # ラベル情報（OriginalFullFaceSub）
├── src/
│   ├── preprocess.py              # データ前処理（FullFace）
│   ├── preprocess_original.py     # データ前処理（OrginalFullFace、顔検出付き）
│   ├── preprocess_original_sub.py # データ前処理（OriginalFullFaceSub、顔検出付き）
│   ├── dataset.py                 # データセットローダー
│   ├── model_ff.py                # モデル定義
│   ├── train.py                   # 学習スクリプト
│   ├── evaluate.py                # 評価スクリプト（train/test分割）
│   └── evaluate_full.py           # 評価スクリプト（全データ使用）
├── configs/
│   └── ff_default.yaml            # デフォルト設定
├── models/
│   ├── best_model.h5              # 学習済みモデル
│   ├── evaluation_original/       # OrginalFullFaceの評価結果
│   └── evaluation_original_sub/   # OriginalFullFaceSubの評価結果
├── requirements.txt               # 依存パッケージ
├── run.sh                         # 一括実行スクリプト（学習）
├── evaluate_original.sh           # OrginalFullFace評価スクリプト
├── evaluate_original_sub.sh       # OriginalFullFaceSub評価スクリプト
└── README.md                      # このファイル
```

## 参考文献

- 論文: [Person-Specific Gaze Estimation from Low-Quality Webcam Images](https://www.mdpi.com/1424-8220/23/8/4138)
- データセット: 論文公開データセット使用

## ライセンス

論文およびデータセットのライセンスに準拠します。
