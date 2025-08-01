#!/bin/bash

# Start Jupyter with covenantv2 environment
# Usage: ./start_jupyter.sh [notebook|lab]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}>>> $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_error() {
    echo -e "${RED}>>> $1${NC}"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found. Please run 'make setup' first."
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Check if ipykernel is installed
if ! python -c "import ipykernel" 2>/dev/null; then
    print_warning "ipykernel not found. Installing..."
    pip install ipykernel
fi

# Install kernel if not already installed
if ! jupyter kernelspec list | grep -q "covenantv2"; then
    print_status "Installing Jupyter kernel for covenantv2 environment..."
    python -m ipykernel install --user --name=covenantv2 --display-name="Covenant v2 Environment"
    print_success "Kernel installed successfully!"
else
    print_success "Kernel already installed."
fi

# Determine which Jupyter to start
JUPYTER_TYPE=${1:-notebook}

case $JUPYTER_TYPE in
    "notebook"|"nb")
        print_status "Starting Jupyter Notebook..."
        print_warning "Make sure to select 'Covenant v2 Environment' kernel in your notebooks"
        jupyter notebook --notebook-dir=.
        ;;
    "lab"|"jupyterlab")
        print_status "Starting Jupyter Lab..."
        print_warning "Make sure to select 'Covenant v2 Environment' kernel in your notebooks"
        jupyter lab --notebook-dir=.
        ;;
    *)
        print_error "Invalid option: $JUPYTER_TYPE"
        echo "Usage: $0 [notebook|lab]"
        echo "  notebook: Start Jupyter Notebook (default)"
        echo "  lab: Start Jupyter Lab"
        exit 1
        ;;
esac 