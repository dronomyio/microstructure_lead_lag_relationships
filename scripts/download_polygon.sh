#!/bin/bash

# Download Polygon quote data
# Requires POLYGON_API_KEY environment variable

API_KEY=${POLYGON_API_KEY:-"demo"}
DATA_DIR="/app/data"
SYMBOLS=("AAPL" "MSFT" "GOOGL")
DATE=$(date -d "yesterday" '+%Y-%m-%d' 2>/dev/null || date -v-1d '+%Y-%m-%d')

mkdir -p ${DATA_DIR}

if [ -z "$API_KEY" ] || [ "$API_KEY" = "demo" ]; then
    echo "Warning: No API key provided, creating demo data files..."
    for SYMBOL in "${SYMBOLS[@]}"; do
        echo '{"results":[]}' > "${DATA_DIR}/${SYMBOL}_demo_quotes.json"
    done
else
    echo "Downloading quotes with API key..."
    for SYMBOL in "${SYMBOLS[@]}"; do
        echo "Downloading quotes for ${SYMBOL} on ${DATE}..."
        curl -X GET "https://api.polygon.io/v3/quotes/${SYMBOL}?date=${DATE}&apiKey=${API_KEY}&limit=50000" \
             -o "${DATA_DIR}/${SYMBOL}_${DATE}_quotes.json" || \
        echo '{"results":[]}' > "${DATA_DIR}/${SYMBOL}_${DATE}_quotes.json"
    done
fi

echo "Data preparation complete!"
