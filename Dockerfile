# High-Frequency Statistical Arbitrage Strategy Development
# Multi-stage Docker build for production deployment

# Stage 1: Base image with system dependencies
FROM ubuntu:20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libboost-all-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA (optional - can be disabled for CPU-only builds)
ARG CUDA_VERSION=11.8
ARG CUDA_AVAILABLE=true
RUN if [ "$CUDA_AVAILABLE" = "true" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
        dpkg -i cuda-keyring_1.0-1_all.deb && \
        apt-get update && \
        apt-get install -y cuda-toolkit-${CUDA_VERSION} && \
        rm -rf /var/lib/apt/lists/* && \
        rm cuda-keyring_1.0-1_all.deb; \
    fi

# Stage 2: Python dependencies
FROM base as python-deps

# Copy requirements file
COPY python/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: C++ build
FROM base as cpp-build

# Copy C++ source code
COPY cpp/ /app/cpp/

# Build C++ components
WORKDIR /app/cpp
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Stage 4: Final production image
FROM base as production

# Copy Python dependencies
COPY --from=python-deps /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy built C++ components
COPY --from=cpp-build /app/cpp/build/market_data_pipeline /usr/local/bin/

# Copy application code
COPY python/ /app/python/
COPY gpu/ /app/gpu/
COPY config/ /app/config/
COPY docs/ /app/docs/
COPY build_and_run.sh /app/
COPY README.md /app/
COPY LICENSE /app/

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p data models logs results

# Set permissions
RUN chmod +x build_and_run.sh

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["./build_and_run.sh"]

# Expose ports for data pipeline
EXPOSE 8888

# Labels
LABEL maintainer="High-Frequency Trading Team"
LABEL version="1.0.0"
LABEL description="High-Frequency Statistical Arbitrage Strategy Development"
LABEL org.opencontainers.image.source="https://github.com/yourusername/high-frequency-arbitrage" 