"""
OriginalFullFaceSubデータの前処理スクリプト
ファイル名から座標ラベルを抽出し、Haar Cascadeで顔検出を行い、224×224にリサイズして保存
"""
import os
import cv2
import pandas as pd
from pathlib import Path
import argparse
import re


# ファイル名から座標ラベルへのマッピング
LABEL_MAP = {
    'topLeft': (20, 20),
    'topRight': (1780, 20),
    'bottomLeft': (20, 1149),
    'bottomRight': (1780, 1149),
    'center': (880, 564.5)
}


def extract_label_from_filename(filename):
    """
    ファイル名から座標ラベルを抽出
    例: topLeft.png -> 'topLeft'
        bottomRight.png -> 'bottomRight'
        center_22.png -> 'center'
        topLeft_22.png -> 'topLeft'
    
    Args:
        filename: ファイル名
        
    Returns:
        label_name: ラベル名（LABEL_MAPのキー）、見つからない場合はNone
    """
    # 拡張子を除去
    name = Path(filename).stem
    
    # アンダースコアで分割（ID付きのファイル名に対応）
    # 例: "center_22" -> ["center", "22"]
    parts = name.split('_')
    
    # LABEL_MAPのキーと照合
    for label_name in LABEL_MAP.keys():
        # 完全一致の場合（例: topLeft.png）
        if label_name.lower() == name.lower():
            return label_name
        # 最初の部分が一致する場合（例: topLeft_22.png）
        if parts[0].lower() == label_name.lower():
            return label_name
    
    return None


def detect_and_crop_face(image, face_cascade):
    """
    Haar Cascadeで顔を検出し、最大の顔領域を切り出す
    
    Args:
        image: 入力画像
        face_cascade: Haar Cascade分類器
        
    Returns:
        face_img: 切り出された顔画像（検出できなければNone）
    """
    # グレースケール化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 顔検出
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None
    
    # 最大の顔を選択
    max_area = 0
    max_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            max_face = (x, y, w, h)
    
    if max_face is None:
        return None
    
    # 顔領域を切り出し
    x, y, w, h = max_face
    face_img = image[y:y+h, x:x+w]
    
    return face_img


def preprocess_original_sub_images(raw_dir, output_dir, labels_csv_path, target_size=(224, 224)):
    """
    OriginalFullFaceSubの画像を前処理して保存し、ラベルCSVを生成
    
    Args:
        raw_dir: 元画像のディレクトリ (OriginalFullFaceSub/)
        output_dir: 処理後画像の保存先
        labels_csv_path: ラベルCSVの保存先
        target_size: リサイズ後のサイズ
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Haar Cascade分類器の読み込み
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
    
    print(f"Haar Cascade loaded from: {cascade_path}")
    print(f"Processing images from {raw_dir}...")
    
    labels = []
    valid_count = 0
    invalid_count = 0
    no_face_count = 0
    
    # ディレクトリ内の全画像を処理
    image_files = list(raw_path.glob("*.png")) + list(raw_path.glob("*.jpg"))
    total = len(image_files)
    
    print(f"\nFound {total} image files")
    print("=" * 60)
    
    for idx, img_path in enumerate(image_files):
        filename = img_path.name
        print(f"\nProcessing [{idx + 1}/{total}]: {filename}")
        
        # ファイル名からラベルを抽出
        label_name = extract_label_from_filename(filename)
        
        if label_name is None:
            print(f"  Warning: Could not extract label from filename {filename}")
            invalid_count += 1
            continue
        
        label_x, label_y = LABEL_MAP[label_name]
        print(f"  Label: {label_name} -> ({label_x}, {label_y})")
        
        # 画像読み込み
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Failed to read {filename}")
            invalid_count += 1
            continue
        
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
        
        # 顔検出
        face_img = detect_and_crop_face(img, face_cascade)
        
        if face_img is None:
            print(f"  Warning: No face detected in {filename}")
            no_face_count += 1
            invalid_count += 1
            continue
        
        print(f"  Face detected: {face_img.shape[1]}x{face_img.shape[0]}")
        
        # 224×224にリサイズ
        face_resized = cv2.resize(face_img, target_size)
        
        # 保存（元のファイル名を維持）
        output_file = output_path / filename
        cv2.imwrite(str(output_file), face_resized)
        print(f"  Saved to: {output_file}")
        
        # ラベル情報を記録
        labels.append({
            'filename': filename,
            'x': label_x,
            'y': label_y,
            'path': str(output_file),
            'label_name': label_name
        })
        valid_count += 1
    
    # ラベルCSVを保存
    df = pd.DataFrame(labels)
    df.to_csv(labels_csv_path, index=False)
    
    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)
    print(f"Valid images (face detected): {valid_count}")
    print(f"No face detected: {no_face_count}")
    print(f"Invalid/skipped images: {invalid_count}")
    print(f"Total processed: {valid_count + invalid_count}")
    print(f"Labels saved to: {labels_csv_path}")
    print(f"Images saved to: {output_dir}")
    print("=" * 60)
    
    # ラベルごとの統計を表示
    print("\nLabel distribution:")
    for label_name, (x, y) in LABEL_MAP.items():
        count = len(df[df['label_name'] == label_name])
        if count > 0:
            print(f"  {label_name:15s} ({x:4.1f}, {y:6.1f}): {count:3d} image(s)")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess OriginalFullFaceSub images with face detection'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='raw_data/OriginalFullFaceSub',
        help='Directory containing raw images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/images_original_sub',
        help='Directory to save processed images'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        default='data/labels_original_sub.csv',
        help='Path to save labels CSV'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Target image size (height width)'
    )
    
    args = parser.parse_args()
    
    preprocess_original_sub_images(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        labels_csv_path=args.labels_csv,
        target_size=tuple(args.target_size)
    )


if __name__ == '__main__':
    main()

