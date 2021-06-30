# Reproducible build + full test run: docker build -t microfeature . && docker run --rm microfeature
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt Makefile ./
RUN make setup
COPY . .
RUN make build
# Default: run the C++ and Python suites, then the ML demo.
CMD ["sh", "-c", "make test && make demo"]
