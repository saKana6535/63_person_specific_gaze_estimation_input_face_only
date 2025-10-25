"""
視線推定モデルの評価スクリプト（全データ使用版）
train/test分割せず、全データを評価に使用
"""
import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def pixel_to_degree(error_px, ppi=96, distance_cm=32.5):
    """
    ピクセル誤差を角度誤差に変換
    
    Args:
        error_px: ピクセル単位の誤差
        ppi: モニタのPPI (pixels per inch)
        distance_cm: 目とスクリーンの距離 (cm)
    
    Returns:
        角度誤差（度）
    """
    # ピクセルをcmに変換
    error_cm = error_px / ppi * 2.54
    
    # 角度に変換（ラジアン -> 度）
    error_rad = np.arctan(error_cm / distance_cm)
    error_deg = np.degrees(error_rad)
    
    return error_deg


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    画像を読み込んで前処理
    
    Args:
        image_path: 画像のパス
        target_size: 目標サイズ
        
    Returns:
        前処理済み画像（正規化済み）
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # RGB変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # リサイズ（念のため）
    img = cv2.resize(img, target_size)
    
    # 正規化 [0, 255] -> [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def evaluate_model_full(model_path, labels_csv_path, batch_size=32, output_dir=None):
    """
    モデルを全データで評価（train/test分割なし）
    
    Args:
        model_path: 学習済みモデルのパス
        labels_csv_path: ラベルCSVのパス
        batch_size: バッチサイズ
        output_dir: 結果の保存先（オプション）
    """
    print("=" * 60)
    print("Model Evaluation (Full Dataset)")
    print("=" * 60)
    
    # モデルの読み込み
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    
    # ラベルCSVの読み込み
    print(f"\nLoading labels from: {labels_csv_path}")
    df = pd.read_csv(labels_csv_path)
    print(f"Total samples: {len(df)}")
    
    # データとラベルを準備
    print("\nLoading images...")
    images = []
    labels = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            img = load_and_preprocess_image(row['path'])
            images.append(img)
            labels.append([row['x'], row['y']])
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Failed to load {row['path']}: {e}")
            continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Successfully loaded {len(images)} images")
    
    # 予測
    print("\nGenerating predictions...")
    predictions = model.predict(images, batch_size=batch_size, verbose=1)
    
    # 評価指標の計算
    print("\nCalculating metrics...")
    
    # 絶対誤差
    errors = np.abs(predictions - labels)
    
    # ユークリッド距離誤差
    errors_euclidean = np.sqrt(np.sum((predictions - labels) ** 2, axis=1))
    
    # 各軸のMAE
    mae_x = np.mean(errors[:, 0])
    mae_y = np.mean(errors[:, 1])
    mae_euclidean = np.mean(errors_euclidean)
    
    # MSE
    mse = np.mean((predictions - labels) ** 2)
    mse_x = np.mean((predictions[:, 0] - labels[:, 0]) ** 2)
    mse_y = np.mean((predictions[:, 1] - labels[:, 1]) ** 2)
    
    # 角度誤差に変換
    error_deg_euclidean = pixel_to_degree(errors_euclidean)
    mean_error_deg = np.mean(error_deg_euclidean)
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Total samples:          {len(images)}")
    print(f"\nMAE (X coordinate):     {mae_x:.4f} px")
    print(f"MAE (Y coordinate):     {mae_y:.4f} px")
    print(f"MAE (Euclidean):        {mae_euclidean:.4f} px")
    print(f"\nMSE (X coordinate):     {mse_x:.4f}")
    print(f"MSE (Y coordinate):     {mse_y:.4f}")
    print(f"MSE (Overall):          {mse:.4f}")
    print(f"\nMean Angular Error:     {mean_error_deg:.4f}°")
    print(f"Max Angular Error:      {np.max(error_deg_euclidean):.4f}°")
    print(f"Min Angular Error:      {np.min(error_deg_euclidean):.4f}°")
    print(f"Std Angular Error:      {np.std(error_deg_euclidean):.4f}°")
    print("=" * 60)
    
    # ラベルごとの誤差統計（original_dirまたはlabel_nameカラムがある場合）
    label_column = None
    if 'original_dir' in df.columns:
        label_column = 'original_dir'
    elif 'label_name' in df.columns:
        label_column = 'label_name'
    
    if label_column:
        print("\n" + "=" * 60)
        print("Error Statistics by Label:")
        print("=" * 60)
        
        df_valid = df.iloc[valid_indices].copy()
        df_valid['error_euclidean'] = errors_euclidean
        df_valid['error_deg'] = error_deg_euclidean
        df_valid['pred_x'] = predictions[:, 0]
        df_valid['pred_y'] = predictions[:, 1]
        
        for label_name in df_valid[label_column].unique():
            mask = df_valid[label_column] == label_name
            label_errors = df_valid[mask]['error_euclidean']
            label_errors_deg = df_valid[mask]['error_deg']
            
            print(f"\n{label_name}:")
            print(f"  Samples: {len(label_errors)}")
            print(f"  MAE (Euclidean): {np.mean(label_errors):.4f} px")
            print(f"  Mean Angular Error: {np.mean(label_errors_deg):.4f}°")
            print(f"  Std Angular Error: {np.std(label_errors_deg):.4f}°")
        
        print("=" * 60)
    
    # 結果を辞書にまとめる
    results = {
        'total_samples': int(len(images)),
        'mae_x': float(mae_x),
        'mae_y': float(mae_y),
        'mae_euclidean': float(mae_euclidean),
        'mse_x': float(mse_x),
        'mse_y': float(mse_y),
        'mse': float(mse),
        'mean_angular_error_deg': float(mean_error_deg),
        'max_angular_error_deg': float(np.max(error_deg_euclidean)),
        'min_angular_error_deg': float(np.min(error_deg_euclidean)),
        'std_angular_error_deg': float(np.std(error_deg_euclidean))
    }
    
    # 結果を保存
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で保存
        results_path = output_path / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        # 予測結果をCSVで保存
        # label_columnを決定
        if 'original_dir' in df.columns:
            label_col_name = 'original_dir'
            label_col_values = df.iloc[valid_indices]['original_dir'].values
        elif 'label_name' in df.columns:
            label_col_name = 'label_name'
            label_col_values = df.iloc[valid_indices]['label_name'].values
        else:
            label_col_name = 'label'
            label_col_values = ['unknown'] * len(valid_indices)
        
        results_df = pd.DataFrame({
            'filename': df.iloc[valid_indices]['filename'].values,
            label_col_name: label_col_values,
            'gt_x': labels[:, 0],
            'gt_y': labels[:, 1],
            'pred_x': predictions[:, 0],
            'pred_y': predictions[:, 1],
            'error_x': errors[:, 0],
            'error_y': errors[:, 1],
            'error_euclidean': errors_euclidean,
            'error_deg': error_deg_euclidean
        })
        csv_path = output_path / 'predictions.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
        
        # 誤差の分布をプロット
        plot_error_distribution(errors_euclidean, error_deg_euclidean, output_path)
        
        # ラベルごとの誤差をプロット（ラベルカラムがある場合）
        if label_col_name in results_df.columns and label_col_name != 'label':
            plot_error_by_label(results_df, output_path, label_col_name)
    
    return results


def plot_error_distribution(errors_px, errors_deg, output_dir):
    """
    誤差分布をプロット
    
    Args:
        errors_px: ピクセル単位の誤差
        errors_deg: 角度単位の誤差
        output_dir: 保存先ディレクトリ
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ピクセル誤差のヒストグラム
    axes[0].hist(errors_px, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors_px), color='red', linestyle='--',
                    label=f'Mean: {np.mean(errors_px):.2f} px')
    axes[0].set_xlabel('Euclidean Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution (Pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 角度誤差のヒストグラム
    axes[1].hist(errors_deg, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(np.mean(errors_deg), color='red', linestyle='--',
                    label=f'Mean: {np.mean(errors_deg):.4f}°')
    axes[1].set_xlabel('Angular Error (degrees)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution (Degrees)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'error_distribution.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Error distribution plot saved to: {plot_path}")


def plot_error_by_label(results_df, output_dir, label_column='original_dir'):
    """
    ラベルごとの誤差をプロット
    
    Args:
        results_df: 予測結果のDataFrame
        output_dir: 保存先ディレクトリ
        label_column: ラベル情報を含むカラム名
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ラベルの順序を固定（統一された順序）
    label_order = ['topLeft', 'topRight', 'center', 'bottomLeft', 'bottomRight']
    
    # データに存在するラベルのみをフィルタ
    available_labels = results_df[label_column].unique()
    labels = [label for label in label_order if label in available_labels]
    
    # ラベルごとのピクセル誤差
    data_px = [results_df[results_df[label_column] == label]['error_euclidean'].values 
               for label in labels]
    axes[0].boxplot(data_px, tick_labels=labels)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Euclidean Error (pixels)')
    axes[0].set_title('Error by Label (Pixels)')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # ラベルごとの角度誤差
    data_deg = [results_df[results_df[label_column] == label]['error_deg'].values 
                for label in labels]
    axes[1].boxplot(data_deg, tick_labels=labels)
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Angular Error (degrees)')
    axes[1].set_title('Error by Label (Degrees)')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'error_by_label.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Error by label plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate gaze estimation model on full dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.h5)'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        required=True,
        help='Path to labels CSV'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for prediction'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    evaluate_model_full(
        model_path=args.model,
        labels_csv_path=args.labels_csv,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

