FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by scipy, scikit-fda, and fdasrsf
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ gfortran pkg-config \
        libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# findblas searches /usr/lib/ but not the ARM64 multiarch path;
# symlink OpenBLAS there and set LIBRARY_PATH so the build can find it
RUN ln -sf /usr/lib/aarch64-linux-gnu/libopenblas.so /usr/lib/libopenblas.so && \
    ln -sf /usr/lib/aarch64-linux-gnu/libopenblas.a /usr/lib/libopenblas.a && \
    ln -sf /usr/lib/aarch64-linux-gnu/liblapack.so /usr/lib/liblapack.so && \
    ln -sf /usr/lib/aarch64-linux-gnu/liblapack.a /usr/lib/liblapack.a
ENV LIBRARY_PATH=/usr/lib/aarch64-linux-gnu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Create output directory
RUN mkdir -p outputs

# Mount points:
#   /data    - host data directory (contains processedjumpdata.mat)
#   /app/outputs - persisted training outputs
VOLUME ["/data", "/app/outputs"]

ENTRYPOINT ["python", "-m", "src.train"]
# Default: show help. Override with your training args.
CMD ["--help"]
