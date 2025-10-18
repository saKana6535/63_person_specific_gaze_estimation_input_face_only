"""
データセットローダー
前処理済み画像とラベルを読み込み、train/testに分割
"""
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf


class GazeDataset:
    """視線推定用データセットクラス"""

    def __init__(self, labels_csv_path, test_size=0.2, random_state=42):
        """
        Args:
            labels_csv_path: ラベルCSVファイルのパス
            test_size: テストデータの割合
            random_state: ランダムシード
        """
        self.df = pd.read_csv(labels_csv_path)
        self.test_size = test_size
        self.random_state = random_state

        # train/test分割
        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state
        )

        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")

    def load_image(self, path):
        """
        画像を読み込んで正規化

        Args:
            path: 画像ファイルのパス

        Returns:
            正規化された画像 (224, 224, 3) float32 [0, 1]
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def create_tf_dataset(self, df, batch_size=32, shuffle=True, augment=False):
        """
        TensorFlow Datasetを作成

        Args:
            df: データフレーム
            batch_size: バッチサイズ
            shuffle: シャッフルするか
            augment: データ拡張を適用するか

        Returns:
            tf.data.Dataset
        """
        def generator():
            indices = np.arange(len(df))
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                row = df.iloc[idx]
                img = self.load_image(row['path'])
                label = np.array([row['x'], row['y']], dtype=np.float32)
                yield img, label

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32)
            )
        )

        if augment:
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _augment(self, image, label):
        """
        データ拡張（ランダム明度調整、コントラスト調整）

        Args:
            image: 入力画像
            label: ラベル

        Returns:
            拡張された画像とラベル
        """
        # ランダム明度調整
        image = tf.image.random_brightness(image, max_delta=0.1)
        # ランダムコントラスト調整
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # [0, 1]にクリップ
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def get_train_dataset(self, batch_size=32, augment=True):
        """学習用データセットを取得"""
        return self.create_tf_dataset(
            self.train_df,
            batch_size=batch_size,
            shuffle=True,
            augment=augment
        )

    def get_test_dataset(self, batch_size=32):
        """テスト用データセットを取得"""
        return self.create_tf_dataset(
            self.test_df,
            batch_size=batch_size,
            shuffle=False,
            augment=False
        )

    def get_stats(self):
        """データセットの統計情報を取得"""
        stats = {
            'total_samples': len(self.df),
            'train_samples': len(self.train_df),
            'test_samples': len(self.test_df),
            'x_mean': self.df['x'].mean(),
            'x_std': self.df['x'].std(),
            'y_mean': self.df['y'].mean(),
            'y_std': self.df['y'].std(),
            'x_min': self.df['x'].min(),
            'x_max': self.df['x'].max(),
            'y_min': self.df['y'].min(),
            'y_max': self.df['y'].max()
        }
        return stats
