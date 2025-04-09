#!/bin/bash

# Smart wrapper for running pipeline locally
# Automatically uses correct Bash version on macOS (for associative arrays to work)

if [[ "$OSTYPE" == "darwin"* ]] && [[ -x "/opt/homebrew/bin/bash" ]]; then
    echo "🧠 Detected macOS with Homebrew Bash... using it to run pipeline"
    /opt/homebrew/bin/bash run_pipeline.sh "$@"
else
    echo "🚀 Using system Bash..."
    bash run_pipeline.sh "$@"
fi
