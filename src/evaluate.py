"""
視線推定モデルの評価スクリプト
MAE（ピクセル）と角度誤差を計算
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

from dataset import GazeDataset


def pixel_to_degree(error_px, ppi=96, distance_cm=60):
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


def evaluate_model(model_path, labels_csv_path, config_path=None, output_dir=None):
    """
    モデルを評価

    Args:
        model_path: 学習済みモデルのパス
        labels_csv_path: ラベルCSVのパス
        config_path: 設定ファイルのパス（オプション）
        output_dir: 結果の保存先（オプション）
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # モデルの読み込み
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 設定ファイルの読み込み（あれば）
    config = None
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        test_size = config['data']['test_size']
        random_state = config['data']['random_state']
        batch_size = config['training']['batch_size']
    else:
        test_size = 0.2
        random_state = 42
        batch_size = 32

    # データセットの読み込み
    print(f"\nLoading dataset from: {labels_csv_path}")
    dataset = GazeDataset(
        labels_csv_path=labels_csv_path,
        test_size=test_size,
        random_state=random_state
    )

    test_ds = dataset.get_test_dataset(batch_size=batch_size)

    # 評価
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_metrics = model.evaluate(test_ds, verbose=1)
    test_loss, test_mae, test_mse = test_metrics

    # 予測値の取得
    print("\nGenerating predictions...")
    predictions = []
    ground_truths = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        predictions.append(preds)
        ground_truths.append(labels.numpy())

    predictions = np.vstack(predictions)
    ground_truths = np.vstack(ground_truths)

    # 詳細な誤差分析
    errors = np.abs(predictions - ground_truths)
    errors_euclidean = np.sqrt(np.sum((predictions - ground_truths) ** 2, axis=1))

    mae_x = np.mean(errors[:, 0])
    mae_y = np.mean(errors[:, 1])
    mae_euclidean = np.mean(errors_euclidean)

    # 角度誤差に変換
    error_deg_euclidean = pixel_to_degree(errors_euclidean)
    mean_error_deg = np.mean(error_deg_euclidean)

    # 結果を表示
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Test Loss (MAE):        {test_loss:.4f} px")
    print(f"Test MAE:               {test_mae:.4f} px")
    print(f"Test MSE:               {test_mse:.4f}")
    print(f"\nMAE (X coordinate):     {mae_x:.4f} px")
    print(f"MAE (Y coordinate):     {mae_y:.4f} px")
    print(f"MAE (Euclidean):        {mae_euclidean:.4f} px")
    print(f"\nMean Angular Error:     {mean_error_deg:.4f}°")
    print(f"Max Angular Error:      {np.max(error_deg_euclidean):.4f}°")
    print(f"Min Angular Error:      {np.min(error_deg_euclidean):.4f}°")
    print(f"Std Angular Error:      {np.std(error_deg_euclidean):.4f}°")
    print("=" * 60)

    # 結果を辞書にまとめる
    results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_mse': float(test_mse),
        'mae_x': float(mae_x),
        'mae_y': float(mae_y),
        'mae_euclidean': float(mae_euclidean),
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
        results_df = pd.DataFrame({
            'gt_x': ground_truths[:, 0],
            'gt_y': ground_truths[:, 1],
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
    axes[0].hist(errors_px, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors_px), color='red', linestyle='--',
                    label=f'Mean: {np.mean(errors_px):.2f} px')
    axes[0].set_xlabel('Euclidean Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution (Pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 角度誤差のヒストグラム
    axes[1].hist(errors_deg, bins=50, edgecolor='black', alpha=0.7, color='orange')
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate gaze estimation model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.h5)')
    parser.add_argument('--labels_csv', type=str, default='data/labels.csv',
                        help='Path to labels CSV')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        labels_csv_path=args.labels_csv,
        config_path=args.config,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
