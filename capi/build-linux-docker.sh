#!/bin/bash
# libdrama_engine.so for Linux x86-64, built in a CUDA 13.3 Ubuntu 22.04 container (glibc 2.35: runs there and on anything
# newer) from the committed HEAD, with machine code for every listed architecture. Its RUNPATH looks beside it and in
# ../higgs-engine, where Dub Studio keeps the CUDA libraries.
#   capi/build-linux-docker.sh <output folder> [cache folder]
set -euo pipefail

ARCHS="sm_75,sm_80,sm_86,sm_89,sm_90,sm_100,sm_120"

if [ "${1:-}" = "--inside" ]; then
  export DEBIAN_FRONTEND=noninteractive CARGO_HOME=/cache/cargo RUSTUP_HOME=/cache/rustup PATH=/cache/cargo/bin:$PATH
  apt-get update -q >/dev/null
  apt-get install -y -q --no-install-recommends curl ca-certificates git pkg-config >/dev/null
  if [ ! -x /cache/cargo/bin/cargo ]; then
    echo "[..] rust stable"
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
  fi
  echo "[..] source $(git -c safe.directory=/src -C /src rev-parse --short HEAD)"
  rm -rf /cache/src && mkdir -p /cache/src
  git -c safe.directory=/src -C /src archive HEAD | tar -x -C /cache/src
  cd /cache/src/capi
  export CUDA_PATH=/usr/local/cuda DRAMA_CUDA_ARCH="$ARCHS" DRAMA_CUDA_NO_PTX=1 CARGO_TARGET_DIR=/cache/target
  export RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN:$ORIGIN/../higgs-engine'
  echo "[..] build ($ARCHS)"
  cargo build --release 2>&1 | tail -5
  cp /cache/target/release/libdrama_engine.so /out/
  readelf -d /out/libdrama_engine.so | grep -E 'NEEDED|RUNPATH'
  sha256sum /out/libdrama_engine.so
  echo "[OK] libdrama_engine.so"
  exit 0
fi

out="${1:?output folder}"
cache="${2:-$HOME/drama-linux2204-cache}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out" "$cache"
cp "$repo/capi/build-linux-docker.sh" "$out/.build-linux-docker.sh"
docker run --rm -v "$repo:/src:ro" -v "$(cd "$out" && pwd):/out" -v "$cache:/cache" nvidia/cuda:13.3.1-devel-ubuntu22.04 bash /out/.build-linux-docker.sh --inside
