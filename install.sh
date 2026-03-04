#!/bin/sh
set -e

# Default settings
REPO="thewulf7/localcode"
BIN_NAME="localcode"
INSTALL_DIR="$HOME/.local/bin"

# Determine OS and Architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

if [ "$OS" = "linux" ]; then
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "x86_64" ]; then
        TARGET="x86_64-unknown-linux-gnu"
    else
        echo "Error: Unsupported architecture '$ARCH' for Linux."
        exit 1
    fi
elif [ "$OS" = "darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        TARGET="aarch64-apple-darwin"
    else
        TARGET="x86_64-apple-darwin"
    fi
else
    echo "Error: Unsupported OS '$OS'."
    exit 1
fi

ASSET_PATTERN="$BIN_NAME-$TARGET.tar.gz"

# Detect releases 
API_URL="https://api.github.com/repos/$REPO/releases/latest"
echo "Fetching latest release from $API_URL..."
DOWNLOAD_URL=$(curl -s $API_URL | grep -E "browser_download_url.*$ASSET_PATTERN" | cut -d '"' -f 4 | head -n 1)

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: Could not find a suitable binary for $TARGET."
    echo "This might mean there isn't a pre-built binary for your platform yet."
    echo "You can still install using cargo: cargo install --git https://github.com/$REPO.git"
    exit 1
fi

# Download Phase
TMP_DIR=$(mktemp -d)
TAR_FILE="$TMP_DIR/localcode.tar.gz"
echo "Downloading $BIN_NAME from $DOWNLOAD_URL..."
curl -L -# -o "$TAR_FILE" "$DOWNLOAD_URL"

# Extract phase
(cd "$TMP_DIR" && tar -xzf localcode.tar.gz)

# Install Phase
echo "Installing $BIN_NAME to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
chmod +x "$TMP_DIR/$BIN_NAME"

mv "$TMP_DIR/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"

rm -rf "$TMP_DIR"

echo "✅ $BIN_NAME installed successfully to $INSTALL_DIR/$BIN_NAME"
echo ""

# Check if INSTALL_DIR is in PATH
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    echo "⚠️  It looks like '$INSTALL_DIR' is not in your PATH."
    
    if [ -c /dev/tty ]; then
        printf "Would you like to automatically add it to your shell profile (e.g. ~/.bashrc)? [y/N] "
        if read -r ans < /dev/tty; then
            if [ "$ans" = "y" ] || [ "$ans" = "Y" ] || [ "$ans" = "yes" ]; then
                ADDED=0
                if [ -f "$HOME/.bashrc" ]; then
                    echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$HOME/.bashrc"
                    echo "✅ Added to ~/.bashrc"
                    ADDED=1
                fi
                if [ -f "$HOME/.zshrc" ]; then
                    echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$HOME/.zshrc"
                    echo "✅ Added to ~/.zshrc"
                    ADDED=1
                fi
                
                if [ $ADDED -eq 1 ]; then
                    echo "Please restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc) to apply."
                else
                    echo "Could not find ~/.bashrc or ~/.zshrc. Please add it manually:"
                    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
                fi
            else
                echo "Please add it manually to your shell profile:"
                echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
            fi
        else
            echo "Please add it manually to your shell profile:"
            echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
        fi
    else
        echo "You may need to add it to your shell profile (e.g., ~/.bashrc, ~/.zshrc):"
        echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    fi
    echo ""
fi

echo "Run '$BIN_NAME --help' to get started."
