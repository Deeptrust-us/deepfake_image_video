#!/bin/bash
# Download Celeb-DF v2 in a screen session

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "Starting Celeb-DF v2 download in screen session 'download'..."
echo ""

# Check if download screen already exists
if screen -list | grep -q "download"; then
    echo "Screen session 'download' already exists!"
    echo "Attach with: screen -r download"
    echo "Or kill it first: screen -X -S download quit"
    exit 1
fi

# Start download in screen session
screen -S download bash -c "cd '$PROJECT_ROOT' && python scripts/download/download_celebdf_complete.py 2>&1 | tee download.log"

echo ""
echo "Download started in screen session 'download'"
echo ""
echo "To attach: screen -r download"
echo "To detach: Press Ctrl+A then D"
echo "To view progress: tail -f download.log"
echo ""
echo "Note: Download is ~9.29GB and may take 30-60 minutes"




