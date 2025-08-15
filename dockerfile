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
    gnuplot \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for visualization
RUN pip3 install matplotlib numpy pandas plotly dash

# Clone and build dependencies
WORKDIR /deps

# RapidJSON
RUN git clone https://github.com/Tencent/rapidjson.git

# Matplot++
RUN git clone https://github.com/alandefreitas/matplotplusplus.git && \
    cd matplotplusplus && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

# Copy project files
WORKDIR /app
COPY . .

# Build the project
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Entry point
CMD ["./build/lead_lag_analyzer"]
