#!/bin/bash
#
# Build an AppImage from a completed cx_Freeze build.
#
# Usage:
#   ./appimage/build-appimage.sh <build-dir> [version] [arch]
#
#   build-dir : directory containing the cx_Freeze output
#               (REAL-Video-Enhancer, lib/, backend/, ...)
#   version   : app version, defaults to 2.4.2
#   arch      : AppImage architecture, defaults to x86_64
#
# The script assembles a squashfs root and runs appimagetool. It works
# locally (FUSE must be available) or inside a container (set FUSE=0).

set -euo pipefail

BUILD_DIR="${1:?Usage: build-appimage.sh <build-dir> [version] [arch]}"
VERSION="${2:-2.4.2}"
ARCH="${3:-x86_64}"
APP_NAME="REAL-Video-Enhancer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
ROOT_DIR="$WORK_DIR/root"
APPIMAGE_OUT="$WORK_DIR/${APP_NAME}-${VERSION}-linux-${ARCH}.AppImage"

trap 'rm -rf "$WORK_DIR"' EXIT

# --- validate build dir -------------------------------------------------
if [ ! -f "$BUILD_DIR/$APP_NAME" ]; then
    echo "ERROR: $BUILD_DIR/$APP_NAME not found. Run the cx_Freeze build first." >&2
    exit 1
fi
if [ ! -f "$BUILD_DIR/backend/rve-backend.py" ]; then
    echo "ERROR: backend/ not found in build dir. Pass --copy_backend to build.py." >&2
    exit 1
fi

# --- assemble squashfs root --------------------------------------------
mkdir -p "$ROOT_DIR"
cp -a "$BUILD_DIR/." "$ROOT_DIR/"
cp "$SCRIPT_DIR/AppRun" "$ROOT_DIR/AppRun"
cp "$SCRIPT_DIR/$APP_NAME.desktop" "$ROOT_DIR/$APP_NAME.desktop"

# Icons: pre-rendered hicolor theme (same artwork as the Flatpak icon).
mkdir -p "$ROOT_DIR/usr/share"
cp -a "$SCRIPT_DIR/icons" "$ROOT_DIR/usr/share/icons"
# Desktop entry references the icon by name; also drop a 512px root icon.
cp "$ROOT_DIR/usr/share/icons/hicolor/512x512/apps/$APP_NAME.png" "$ROOT_DIR/$APP_NAME.png"
chmod +x "$ROOT_DIR/AppRun"

# --- locate appimagetool (no distro dependency) -------------------------
APPIMAGETOOL="$(command -v appimagetool || true)"
if [ -z "$APPIMAGETOOL" ]; then
    echo "appimagetool not found, downloading..."
    APPIMAGETOOL="$WORK_DIR/appimagetool.AppImage"
    curl -sL -o "$APPIMAGETOOL" \
        "https://github.com/AppImage/appimagetool/releases/latest/download/appimagetool-${ARCH}.AppImage"
    chmod +x "$APPIMAGETOOL"
fi

# --- create the AppImage ------------------------------------------------
# Name/version come from the .desktop file; arch from the ARCH env var.
echo "Building AppImage (${VERSION}, ${ARCH})..."
ARCH="$ARCH" "$APPIMAGETOOL" "$ROOT_DIR" "$APPIMAGE_OUT"

# --- final output -------------------------------------------------------
FINAL="$SCRIPT_DIR/${APP_NAME}-${VERSION}-linux-${ARCH}.AppImage"
mv "$APPIMAGE_OUT" "$FINAL"
chmod +x "$FINAL"
echo "Done: $FINAL"
