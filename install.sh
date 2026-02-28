#!/bin/sh
set -e

# Default settings
REPO="AlexsJones/localcode"
BIN_NAME="localcode"
INSTALL_DIR="/usr/local/bin"

# Determine OS and Architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

if [ "$OS" = "linux" ]; then
    OS_NAME="ubuntu"
elif [ "$OS" = "darwin" ]; then
    OS_NAME="macos"
else
    echo "Error: Unsupported OS '$OS'."
    exit 1
fi

# Detect releases 
API_URL="https://api.github.com/repos/$REPO/releases/latest"
echo "Fetching latest release from $API_URL..."
DOWNLOAD_URL=$(curl -s $API_URL | grep "browser_download_url.*$BIN_NAME-$OS_NAME" | cut -d '"' -f 4 | head -n 1)

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: Could not find a suitable binary for $OS_NAME."
    echo "This might mean there isn't a pre-built binary for your platform yet."
    echo "You can still install using cargo: cargo install --git https://github.com/$REPO.git"
    exit 1
fi

# Download Phase
TMP_FILE=$(mktemp)
echo "Downloading $BIN_NAME from $DOWNLOAD_URL..."
curl -L -# -o "$TMP_FILE" "$DOWNLOAD_URL"

# Install Phase
echo "Installing $BIN_NAME to $INSTALL_DIR..."
chmod +x "$TMP_FILE"

if [ -w "$INSTALL_DIR" ]; then
    mv "$TMP_FILE" "$INSTALL_DIR/$BIN_NAME"
else
    echo "Requires sudo privileges to write to $INSTALL_DIR"
    sudo mv "$TMP_FILE" "$INSTALL_DIR/$BIN_NAME"
fi

echo "✅ $BIN_NAME installed successfully to $INSTALL_DIR/$BIN_NAME"
echo ""
echo "Run '$BIN_NAME --help' to get started."
