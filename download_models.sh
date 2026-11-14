#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-./models}"
mkdir -p "$DEST_DIR"

download_and_verify() {
  local filename="$1"
  local expected_sha="$2"
  local url="$3"
  local dest="$DEST_DIR/$filename"

  if [[ -f "$dest" ]]; then
    local existing_sha
    existing_sha="$(shasum -a 256 "$dest" 2>/dev/null | awk '{print $1}' || sha256sum "$dest" | awk '{print $1}')"
    if [[ "$existing_sha" == "$expected_sha" ]]; then
      echo "[ok] $filename already verified"
      return 0
    fi
  fi

  echo "[download] $url -> $dest"
  curl -fsSL "$url" -o "$dest.tmp"
  local actual_sha
  actual_sha="$(shasum -a 256 "$dest.tmp" 2>/dev/null | awk '{print $1}' || sha256sum "$dest.tmp" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    rm -f "$dest.tmp"
    echo "[error] SHA-256 mismatch for $filename: expected $expected_sha, got $actual_sha" >&2
    exit 1
  fi
  mv "$dest.tmp" "$dest"
  echo "[ok] $filename ($actual_sha)"
}

download_and_verify \
  "4x_nadeshiko_dat2_v1.safetensors" \
  "5f7361323c50b869683cfe7799209600a509d81374bd59e31e5a4a68ab2ae702" \
  "https://github.com/EmilyTsum/nadeshiko-sr/releases/download/v1.0.0/4x_nadeshiko_dat2_v1.safetensors"

download_and_verify \
  "2x_nadeshiko_fdat_v1.safetensors" \
  "7bb87e04b1fe638deca8a5b36439490ce7764c256398c83342fec60aac0a2305" \
  "https://github.com/EmilyTsum/nadeshiko-sr/releases/download/v1.0.0/2x_nadeshiko_fdat_v1.safetensors"
