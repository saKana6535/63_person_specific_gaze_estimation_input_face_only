"""
視線推定モデルの学習スクリプト
"""
import os
import argparse
import yaml
from pathlib import Path
import tensorflow as tf
from datetime import datetime
import json

from dataset import GazeDataset
from model_ff import create_model


def train(config_path):
    """
    モデルを学習

    Args:
        config_path: 設定ファイルのパス
    """
    # 設定ファイルを読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Starting training with configuration:")
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)

    # GPUメモリの動的確保設定
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")

    # データセットの読み込み
    print("\nLoading dataset...")
    dataset = GazeDataset(
        labels_csv_path=config['data']['labels_csv'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # データセットの統計情報を表示
    stats = dataset.get_stats()
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    # train/testデータセットの作成
    train_ds = dataset.get_train_dataset(
        batch_size=config['training']['batch_size'],
        augment=config['data'].get('augment', True)
    )
    test_ds = dataset.get_test_dataset(
        batch_size=config['training']['batch_size']
    )

    # モデルの作成
    print("\nBuilding model...")
    model = create_model(config['model'])
    model.summary()

    # 保存先ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config['training']['save_dir']) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # 設定ファイルを保存
    config_save_path = save_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nConfiguration saved to: {config_save_path}")

    # コールバックの設定
    callbacks = []

    # ModelCheckpoint
    checkpoint_path = save_dir / 'best_model.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_mae',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)

    # EarlyStopping
    if config['training'].get('early_stopping', True):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=config['training'].get('patience', 10),
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)

    # CSVLogger
    csv_path = save_dir / 'training_log.csv'
    csv_logger = tf.keras.callbacks.CSVLogger(str(csv_path))
    callbacks.append(csv_logger)

    # TensorBoard
    tensorboard_dir = save_dir / 'tensorboard'
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1
    )
    callbacks.append(tensorboard)

    # 学習開始
    print("\n" + "=" * 60)
    print("Training started...")
    print("=" * 60 + "\n")

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # 最終モデルを保存
    final_model_path = save_dir / 'final_model.h5'
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    # 学習履歴を保存
    history_path = save_dir / 'history.json'
    with open(history_path, 'w') as f:
        history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # 最終評価
    print("\n" + "=" * 60)
    print("Final evaluation on test set:")
    print("=" * 60)
    test_metrics = model.evaluate(test_ds, verbose=1)

    metrics_dict = {
        'test_loss': float(test_metrics[0]),
        'test_mae': float(test_metrics[1]),
        'test_mse': float(test_metrics[2])
    }

    # メトリクスを保存
    metrics_path = save_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nTest metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)

    return model, history, save_dir


def main():
    parser = argparse.ArgumentParser(description='Train gaze estimation model')
    parser.add_argument('--config', type=str, default='configs/ff_default.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    train(args.config)


if __name__ == '__main__':
    main()
