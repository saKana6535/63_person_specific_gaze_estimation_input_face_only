"""
データ前処理スクリプト
ファイル名から座標を抽出し、画像を224×224にリサイズして保存
"""
import os
import re
import cv2
import pandas as pd
from pathlib import Path
import argparse


def parse_label_from_filename(filename):
    """
    ファイル名から (x, y) 座標を抽出
    例: face_1645_1064_3.jpg -> (1645, 1064)
    """
    match = re.match(r"face_(\d+)_(\d+)_\d+", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None


def preprocess_images(raw_dir, output_dir, labels_csv_path, target_size=(224, 224)):
    """
    画像を前処理して保存し、ラベルCSVを生成

    Args:
        raw_dir: 元画像のディレクトリ
        output_dir: 処理後画像の保存先
        labels_csv_path: ラベルCSVの保存先
        target_size: リサイズ後のサイズ
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = []
    valid_count = 0
    invalid_count = 0

    print(f"Processing images from {raw_dir}...")

    # 全画像を処理
    image_files = list(raw_path.glob("*.jpg"))
    total = len(image_files)

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 1000 == 0:
            print(f"Progress: {idx + 1}/{total}")

        filename = img_path.name
        x, y = parse_label_from_filename(filename)

        if x is None or y is None:
            invalid_count += 1
            continue

        # 画像読み込み
        img = cv2.imread(str(img_path))
        if img is None:
            invalid_count += 1
            continue

        # 224×224にリサイズ
        img_resized = cv2.resize(img, target_size)

        # 保存
        output_file = output_path / filename
        cv2.imwrite(str(output_file), img_resized)

        # ラベル情報を記録
        labels.append({
            'filename': filename,
            'x': x,
            'y': y,
            'path': str(output_file)
        })
        valid_count += 1

    # ラベルCSVを保存
    df = pd.DataFrame(labels)
    df.to_csv(labels_csv_path, index=False)

    print(f"\nPreprocessing completed!")
    print(f"Valid images: {valid_count}")
    print(f"Invalid/skipped images: {invalid_count}")
    print(f"Labels saved to: {labels_csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Preprocess face images for gaze estimation')
    parser.add_argument('--raw_dir', type=str, default='raw_data/FullFace',
                        help='Directory containing raw images')
    parser.add_argument('--output_dir', type=str, default='data/images',
                        help='Directory to save processed images')
    parser.add_argument('--labels_csv', type=str, default='data/labels.csv',
                        help='Path to save labels CSV')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (height width)')

    args = parser.parse_args()

    preprocess_images(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        labels_csv_path=args.labels_csv,
        target_size=tuple(args.target_size)
    )


if __name__ == '__main__':
    main()
