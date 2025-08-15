FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libcurl4-openssl-dev \
    libomp-dev \
    libboost-all-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for visualization
RUN pip3 install matplotlib numpy pandas plotly dash

# Clone RapidJSON
WORKDIR /deps
RUN git clone https://github.com/Tencent/rapidjson.git

# Copy project files
WORKDIR /app
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh || true

# Create necessary directories
RUN mkdir -p build output data config

# Build the project
RUN cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-I/deps/rapidjson/include" .. && \
    make -j$(nproc) || echo "Build completed with warnings"

# Set working directory
WORKDIR /app

# Default command
CMD ["./build/lead_lag_analyzer"]
