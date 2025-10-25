"""
OrginalFullFaceデータの前処理スクリプト
Haar Cascadeで顔検出を行い、224×224にリサイズして保存
"""
import os
import cv2
import pandas as pd
from pathlib import Path
import argparse


# ディレクトリ名から座標ラベルへのマッピング
LABEL_MAP = {
    'topLeft': (20, 20),
    'topRight': (1780, 20),
    'bottomLeft': (20, 1149),
    'bottomRight': (1780, 1149),
    'center': (880, 564.5)
}


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


def preprocess_original_images(raw_dir, output_dir, labels_csv_path, target_size=(224, 224)):
    """
    OrginalFullFaceの画像を前処理して保存し、ラベルCSVを生成
    
    Args:
        raw_dir: 元画像のディレクトリ (OrginalFullFace/)
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
    
    # 各ディレクトリを処理
    for dir_name, (label_x, label_y) in LABEL_MAP.items():
        dir_path = raw_path / dir_name
        
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist. Skipping...")
            continue
        
        print(f"\nProcessing directory: {dir_name} (label: {label_x}, {label_y})")
        
        # ディレクトリ内の全画像を処理
        image_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg"))
        total = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{total}")
            
            filename = img_path.name
            
            # 画像読み込み
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: Failed to read {filename}")
                invalid_count += 1
                continue
            
            # 顔検出
            face_img = detect_and_crop_face(img, face_cascade)
            
            if face_img is None:
                print(f"  Warning: No face detected in {filename}")
                no_face_count += 1
                invalid_count += 1
                continue
            
            # 224×224にリサイズ
            face_resized = cv2.resize(face_img, target_size)
            
            # 保存（元のファイル名を維持）
            output_file = output_path / f"{dir_name}_{filename}"
            cv2.imwrite(str(output_file), face_resized)
            
            # ラベル情報を記録
            labels.append({
                'filename': f"{dir_name}_{filename}",
                'x': label_x,
                'y': label_y,
                'path': str(output_file),
                'original_dir': dir_name
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
    for dir_name, (x, y) in LABEL_MAP.items():
        count = len(df[df['original_dir'] == dir_name])
        print(f"  {dir_name:15s} ({x:4.1f}, {y:6.1f}): {count:3d} images")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess OrginalFullFace images with face detection'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='raw_data/OrginalFullFace',
        help='Directory containing raw images (with subdirectories)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/images_original',
        help='Directory to save processed images'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        default='data/labels_original.csv',
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
    
    preprocess_original_images(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        labels_csv_path=args.labels_csv,
        target_size=tuple(args.target_size)
    )


if __name__ == '__main__':
    main()





