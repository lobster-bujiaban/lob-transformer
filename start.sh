#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
  echo "错误：未找到 uv，请先安装 uv 并确保其位于 PATH 中。" >&2
  exit 1
fi

# 后面的命令行参数可覆盖默认 checkpoint、host 和 port。
exec uv run lob-transformer serve --checkpoint model.npz --host 127.0.0.1 --port 8000 "$@"
