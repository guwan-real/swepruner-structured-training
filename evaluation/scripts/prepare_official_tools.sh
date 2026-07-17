#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="${1:-/home/yuantao/futao/swepruner_workspace/official_eval}"
SWE_PRUNER_REF="96171b5f3ecaf89745cbeb436c8893b57f3400bd"
SWE_PRUNER_DIR="$TOOLS_DIR/swe-pruner"

clone_at() {
  local url="$1"
  local destination="$2"
  local revision="$3"
  if [[ ! -d "$destination/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$url" "$destination"
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach "$revision"
}

mkdir -p "$TOOLS_DIR"
clone_at https://github.com/Ayanami1314/swe-pruner.git "$SWE_PRUNER_DIR" "$SWE_PRUNER_REF"

python -m pip install -e "$SWE_PRUNER_DIR/swe-pruner"

cat > "$TOOLS_DIR/tool_revisions.txt" <<EOF
swe-pruner=$SWE_PRUNER_REF
EOF

echo "Official evaluation tools are ready under $TOOLS_DIR"
