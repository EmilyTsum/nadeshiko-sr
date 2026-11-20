# nadeshiko-sr

ミリシタのテクスチャ用超解像モデルと推論環境です。

![preview](preview.png)

| 倍率 | ファイル名 | SHA-256 |
| :--- | :--- | :--- |
| 4x | `4x_nadeshiko_dat2_v1.safetensors` | `5f7361323c50b869683cfe7799209600a509d81374bd59e31e5a4a68ab2ae702` |
| 2x | `2x_nadeshiko_fdat_v1.safetensors` | `7bb87e04b1fe638deca8a5b36439490ce7764c256398c83342fec60aac0a2305` |

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p ./models
# モデルは Releases (v1.0.0) からダウンロードして ./models に配置
# https://github.com/EmilyTsum/nadeshiko-sr/releases/tag/v1.0.0
curl -L -o ./models/4x_nadeshiko_dat2_v1.safetensors https://github.com/EmilyTsum/nadeshiko-sr/releases/download/v1.0.0/4x_nadeshiko_dat2_v1.safetensors
curl -L -o ./models/2x_nadeshiko_fdat_v1.safetensors https://github.com/EmilyTsum/nadeshiko-sr/releases/download/v1.0.0/2x_nadeshiko_fdat_v1.safetensors
```

## 実行

```bash
python3 infer.py \
  --model ./models/2x_nadeshiko_fdat_v1.safetensors \
  --input input.png \
  --output output.png
```
