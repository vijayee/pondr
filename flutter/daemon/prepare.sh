#!/usr/bin/env bash
# prepare.sh — the FFI binding's dev-run one command (the plan's Task 6):
# build SecretAgent's shared client + daemon, drop both where the Dart side
# finds them, print the optional SA_LIBRARY_PATH steer.
#
#   cd pondr/flutter
#   flutter pub get
#   daemon/prepare.sh
#   flutter run --dart-define=pondr.mode=daemon
#
# WHERE THE PIECES LAND (the loader's rules — lib/ffi/sa_ffi.dart's
# resolveSaLibraryPath: the SA_LIBRARY_PATH define, then the SA_LIBRARY_PATH
# env, then `$ExeDir/lib/libsa_client.so`, then `./libsa_client.so`):
#   - flutter/libsa_client.so      — the `./libsa_client.so` candidate; a
#     `flutter run` from the project dir resolves it directly.
#   - build/linux/{debug,release}/bundle/lib/libsa_client.so — the
#     `$ExeDir/lib` candidate, for a built bundle that already exists.
#   - flutter/daemon/frame-demo    — the supervisor's default binary
#     (lib/data/bindings.dart resolves `daemon/frame-demo` relative to the
#     run's working directory, then the PATH's bare name).
# In a shell that wants the explicit steer:
#   export SA_LIBRARY_PATH="$(pwd)/libsa_client.so"
#
# Idempotent: rebuild always (cmake is incremental); the copies are plain
# cp-over (a same-content copy is a no-op touch at the OS level — the
# watcher-relevant file remains the flutter tree's own).
set -euo pipefail

# The script lives at <pondr>/flutter/daemon/prepare.sh; SecretAgent is the
# sibling of pondr in this workspace layout (overridable with
# SECRETAGENT_ROOT for a foreign checkout).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUTTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SA_ROOT="${SECRETAGENT_ROOT:-$(cd "$FLUTTER_DIR/../../SecretAgent" && pwd)}"
BUILD_DIR="$SA_ROOT/cmake-build-debug"

echo "==> SecretAgent root: $SA_ROOT"
if [ ! -d "$SA_ROOT" ]; then
  echo "!! SecretAgent root not found — set SECRETAGENT_ROOT=<path>" >&2
  exit 1
fi

# The debug configure (first run only — the cache marks it).
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "==> configure (cmake -S $SA_ROOT -B $BUILD_DIR)"
  cmake -S "$SA_ROOT" -B "$BUILD_DIR"
fi

echo "==> build (sa_client + frame-demo)"
cmake --build "$BUILD_DIR" -j --target sa_client frame-demo

SO="$BUILD_DIR/libsa_client.so"
DEMO="$BUILD_DIR/frame-demo"
for artifact in "$SO" "$DEMO"; do
  if [ ! -f "$artifact" ]; then
    echo "!! build did not produce $artifact" >&2
    exit 1
  fi
done

echo "==> copy $SO"
# 1. the run-cwd candidate (the dev `flutter run` from flutter/)
cp -f "$SO" "$FLUTTER_DIR/libsa_client.so"
# 2. every existing linux bundle's lib/ (the exe-dir candidate; nothing to
#    do before a first build)
for bundle in "$FLUTTER_DIR"/build/linux/*/*/bundle/lib; do
  if [ -d "$bundle" ]; then
    cp -f "$SO" "$bundle/libsa_client.so"
    echo "    -> $bundle/libsa_client.so"
  fi
done

echo "==> copy $DEMO"
mkdir -p "$FLUTTER_DIR/daemon"
cp -f "$DEMO" "$FLUTTER_DIR/daemon/frame-demo"

echo
echo "==> done. the dev run:"
echo "      flutter run --dart-define=pondr.mode=daemon"
echo "    (optional explicit steer, in this shell first:)"
echo "      export SA_LIBRARY_PATH='$FLUTTER_DIR/libsa_client.so'"