"""
顔画像のみを用いた視線推定モデル (FullFace Model)
論文の構成に基づいたCNNモデル
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_ff_model(input_shape=(224, 224, 3), kernel_size=7, dropout_rate=0.1):
    """
    顔画像から視線座標を推定するCNNモデルを構築

    Args:
        input_shape: 入力画像のサイズ (height, width, channels)
        kernel_size: 畳み込み層のカーネルサイズ
        dropout_rate: Dropoutの割合

    Returns:
        Kerasモデル
    """
    # 入力層
    x_in = layers.Input(shape=input_shape, name='input_face')

    # Conv Block 1
    x = layers.Conv2D(32, kernel_size, padding='same', activation='relu', name='conv1')(x_in)

    # Conv Block 2
    x = layers.Conv2D(96, kernel_size, padding='same', activation='relu', name='conv2')(x)

    # MaxPooling + BatchNormalization
    x = layers.MaxPooling2D(pool_size=2, name='pool1')(x)
    x = layers.BatchNormalization(name='bn1')(x)

    # Conv Block 3
    x = layers.Conv2D(160, kernel_size, padding='same', activation='relu', name='conv3')(x)

    # BatchNormalization
    x = layers.BatchNormalization(name='bn2')(x)

    # Flatten
    x = layers.Flatten(name='flatten')(x)

    # Dense Block
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Output Layer (2次元座標)
    out = layers.Dense(2, activation='linear', name='output_gaze')(x)

    # モデル作成
    model = models.Model(inputs=x_in, outputs=out, name='FullFaceGazeModel')

    return model


def create_model(config):
    """
    設定ファイルからモデルを作成

    Args:
        config: 設定辞書

    Returns:
        コンパイル済みKerasモデル
    """
    model = build_ff_model(
        input_shape=tuple(config.get('input_shape', [224, 224, 3])),
        kernel_size=config.get('kernel_size', 7),
        dropout_rate=config.get('dropout_rate', 0.1)
    )

    # コンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4)),
        loss='mae',
        metrics=['mae', 'mse']
    )

    return model


if __name__ == '__main__':
    # モデルのテスト
    model = build_ff_model()
    model.summary()

    # パラメータ数の確認
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
