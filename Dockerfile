ARG PREPARED_IMAGE=local-semantic-atlas-prepared:0.1.0
FROM ${PREPARED_IMAGE} AS builder
USER 0:0
COPY . /build-source/
WORKDIR /build-source
RUN python -m build --no-isolation --sdist --wheel && \
    python -I qualification/check_artifacts.py /build-source /build-source/dist /opt/artifacts

FROM python@sha256:281d39ff02e3b389401586134ff9e6572954687822a58055ec07a60ded4e0b7f
COPY --from=builder /opt/wheelhouse/ /opt/wheelhouse/
COPY --from=builder /opt/artifacts/ /opt/artifacts/
COPY requirements-runtime.lock /opt/requirements-runtime.lock
RUN python -m pip install --no-cache-dir --no-index --find-links=/opt/wheelhouse --require-hashes -r /opt/requirements-runtime.lock && \
    python -m pip install --no-cache-dir --no-index --no-deps /opt/artifacts/from-sdist/local_semantic_atlas-0.1.0-py3-none-any.whl && \
    python -m pip check && rm -rf /opt/wheelhouse && mkdir -p /work && chown 1000:1000 /work
COPY --from=builder /work/model/ /opt/model/
COPY tests/ /opt/tests/
COPY qualification/native_integration.py qualification/native_cli.py qualification/check_installed.py /opt/checks/
COPY examples/ /opt/examples/
ENV HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
WORKDIR /work
CMD ["/bin/sh", "-c", "python -I /opt/checks/check_installed.py && python -I -B -m unittest discover -s /opt/tests -v && python -I /opt/checks/native_integration.py /opt/model && python -I /opt/checks/native_cli.py"]
