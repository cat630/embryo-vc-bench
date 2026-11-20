# Dockerfile -- optional Linux reproduction path for embryo-vc-bench.
#
# The analyses in this article were executed on Windows 11 / Python 3.13.14 with
# TWO deliberately isolated virtual environments. This file reproduces that
# split inside one image. It is provided as an interoperability aid; it has NOT
# been CI-tested on Linux and is not claimed as the environment in which the
# published numbers were produced (see release/environment/platform.json).
#
# Build (from the repository root):
#   docker build -f release/environment/Dockerfile -t embryo-vc-bench .
# Run the read-only replay (no model loading, no rescoring):
#   docker run --rm -v "$PWD":/app/repo embryo-vc-bench \
#       /opt/venv/bin/python /app/repo/release/reproduce.py
#
# Known portability caveats, recorded honestly:
#   * The two lock files were produced by `pip freeze` on Windows/CPython 3.13.
#     Torch is pinned to the CPU build 2.15.0+cpu; on Linux resolve the
#     equivalent wheel from the PyTorch CPU index before installing.
#   * scgpt==0.2.4 is published only on the official PyPI index and must be
#     installed with --no-deps into /opt/venv-models, because its dependency
#     pins downgrade scvi-tools in the main environment.
#   * flash-attn / tdigest C extensions and torchtext do not build on Python
#     3.13. The repository supplies namespace stubs and pure-PyTorch
#     equivalents (see w5_adapters.py); no compiled extension is required.
FROM python:3.13.14-slim

WORKDIR /app/repo
COPY release/environment/requirements-main.lock.txt /app/locks/requirements-main.lock.txt
COPY release/environment/requirements-models.lock.txt /app/locks/requirements-models.lock.txt
COPY release/environment/environment.yml /app/locks/environment.yml
COPY release /app/repo/release
COPY results /app/repo/results
COPY data_manifest.csv model_manifest.csv README.md /app/repo/

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r /app/locks/requirements-main.lock.txt

RUN python -m venv /opt/venv-models \
 && /opt/venv-models/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv-models/bin/pip install --no-cache-dir -r /app/locks/requirements-models.lock.txt

ENV PATH="/opt/venv/bin:${PATH}"
ENV EMBRYO_VC_BENCH_MODELS_PYTHON="/opt/venv-models/bin/python"
CMD ["/opt/venv/bin/python", "/app/repo/release/reproduce.py"]
