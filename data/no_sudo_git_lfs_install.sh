# Set install directory
INSTALL_DIR="$HOME/bin"
mkdir -p "$INSTALL_DIR"

# Get latest Git LFS release download URL
LATEST_LFS_URL=$(curl -s https://api.github.com/repos/git-lfs/git-lfs/releases/latest | \
                 grep "browser_download_url.*linux-amd64" | \
                 cut -d '"' -f 4)

# Download and extract
curl -L "$LATEST_LFS_URL" -o "$INSTALL_DIR/git-lfs.tar.gz"
tar -xvzf "$INSTALL_DIR/git-lfs.tar.gz" -C "$INSTALL_DIR" --strip-components=1
rm "$INSTALL_DIR/git-lfs.tar.gz"

# Add to PATH (only if not already added)
if ! grep -q 'export PATH=$HOME/bin:$PATH' "$HOME/.bashrc"; then
    echo 'export PATH=$HOME/bin:$PATH' >> "$HOME/.bashrc"
    echo 'export PATH=$HOME/bin:$PATH' >> "$HOME/.bash_profile"
fi

# Apply new PATH
export PATH="$HOME/bin:$PATH"

# Initialize Git LFS
git lfs install --skip-repo

# Verify installation
git lfs version
