# Reference legacy runtime, NOT a hardened or build-verified production image.
FROM python:3.6.15-buster
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/vendor/OmniAnomaly \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir pip==21.3.1 setuptools==59.6.0 wheel==0.37.1 \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip check
COPY pad_omni /app/pad_omni
COPY vendor /app/vendor
COPY config.yaml app.py /app/
COPY tests /app/tests
COPY scripts /app/scripts
RUN mkdir -p /data/models && chgrp -R 0 /app /data && chmod -R g=u /app /data
USER 1001
EXPOSE 8080
CMD ["python", "app.py", "--config", "/app/config.yaml"]
