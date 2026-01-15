#!/bin/bash
# Quick setup script for Data Quality Research

echo "================================"
echo "Data Quality Research - Setup"
echo "================================"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run verification
echo "Running verification tests..."
python test_setup.py

echo ""
echo "Setup complete! Next steps:"
echo "  1. Activate environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "  2. Run experiments: python run_experiments.py --sentence-only"
