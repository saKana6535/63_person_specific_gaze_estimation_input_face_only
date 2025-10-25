# GPU環境セットアップガイド

## 前提条件

- NVIDIA GPU（CUDA対応）
- Ubuntu 20.04以上 または 互換性のあるLinuxディストリビューション
- **Python 3.8以上**（TensorFlow 2.13の場合は3.8-3.11、TensorFlow 2.15以降は3.9以上）

## 1. CUDA Toolkitのインストール

### Python 3.8の場合（TensorFlow 2.12）

TensorFlow 2.12は**CUDA 11.2**が必要です。

```bash
# CUDA 11.2のインストール（Ubuntu）
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
sudo sh cuda_11.2.2_460.32.03_linux.run

# 環境変数の設定
echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# CUDAバージョン確認
nvcc --version
```

### Python 3.9以上の場合（TensorFlow 2.15+）

TensorFlow 2.15以降は**CUDA 12.x**が必要です。

```bash
# CUDA 12.6のインストール例（Ubuntu）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# 環境変数の設定
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# CUDAバージョン確認
nvcc --version
```

## 2. cuDNNのインストール

### Python 3.8の場合（TensorFlow 2.12 + CUDA 11.2）

TensorFlow 2.12は**cuDNN 8.1**が必要です。

```bash
# cuDNN 8.1のインストール
sudo apt-get install libcudnn8=8.1.1.33-1+cuda11.2
sudo apt-get install libcudnn8-dev=8.1.1.33-1+cuda11.2

# cuDNNバージョン確認
ldconfig -p | grep cudnn
```

**注意**: TensorFlow 2.11以降は`tensorflow-gpu`パッケージが廃止され、`tensorflow`パッケージに統合されています。GPU版も`tensorflow`でインストールします。

### Python 3.9以上の場合（TensorFlow 2.15+ + CUDA 12.x）

TensorFlow 2.15以降は**cuDNN 8.9以上**が必要です。

```bash
# cuDNN 9.x のインストール（推奨）
sudo apt-get install libcudnn9-cuda-12
sudo apt-get install libcudnn9-dev-cuda-12

# cuDNNバージョン確認
ldconfig -p | grep cudnn
```

## 3. Python環境のセットアップ

### Python 3.8の場合（TensorFlow 2.12）

```bash
# 現在のPythonバージョン確認
python --version  # Python 3.8.x

# 仮想環境の作成
python -m venv venv
source venv/bin/activate

# pipのアップグレード
pip install --upgrade pip

# GPU版パッケージのインストール（Python 3.8用）
pip install -r requirements_gpu_py38.txt
```

**重要**: TensorFlow 2.11以降は`tensorflow`パッケージがGPU対応を含んでいます。`tensorflow-gpu`パッケージは不要です。

### Python 3.9以上の場合（TensorFlow 2.15+）

```bash
# pyenvでPython 3.10を設定
pyenv local 3.10.13

# 仮想環境の作成
python -m venv venv
source venv/bin/activate

# pipのアップグレード
pip install --upgrade pip

# GPU版パッケージのインストール
pip install -r requirements_gpu.txt
```

## 4. GPU認識の確認

```bash
# GPUデバイスの確認
nvidia-smi

# TensorFlowからのGPU認識確認
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

期待される出力：
```
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 5. トラブルシューティング

### GPUが認識されない場合

**症状:**
```
Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly
```

**対処法:**

#### A. LD_LIBRARY_PATHの確認
```bash
echo $LD_LIBRARY_PATH
# /usr/local/cuda-12.6/lib64が含まれているか確認

# 含まれていない場合
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

#### B. CUDAライブラリの確認
```bash
# 必要なライブラリが存在するか確認
ls /usr/local/cuda-12.6/lib64/libcudart.so*
ls /usr/local/cuda-12.6/lib64/libcublas.so*
ls /usr/lib/x86_64-linux-gnu/libcudnn.so*
```

#### C. 互換性のあるバージョンの再インストール

TensorFlow 2.15系を使用する場合（CUDA 11.8 + cuDNN 8.6でも動作）:

```bash
pip uninstall tensorflow
pip install tensorflow==2.15.0
```

その場合、以下もインストール：
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# cuDNN 8.6
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8
```

#### D. TensorRTの警告（無視可能）
```
TF-TRT Warning: Could not find TensorRT
```
これは警告であり、GPUの基本的な動作には影響しません。必要に応じて以下でインストール：
```bash
pip install tensorrt
```

## 6. 学習の実行

GPU環境が正しく設定されたら、通常通り学習を実行：

```bash
source venv/bin/activate
./run.sh
```

または：

```bash
python src/train.py --config configs/ff_default.yaml
```

学習開始時に以下のメッセージが表示されればGPUが使用されています：
```
GPUs available: 1
```

## 参考リンク

- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
