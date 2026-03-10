#!/bin/bash
# Run Vietnamese AI Call Center Demo

# Change to project directory
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements-app.txt
fi

# Run the demo
echo "Starting Vietnamese AI Call Center Demo..."
echo "Open http://localhost:8501 in your browser"
echo ""

streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
