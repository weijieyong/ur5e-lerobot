FROM ubuntu:24.04

ARG PIXI_VERSION=0.83.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL https://pixi.sh/install.sh -o /tmp/install-pixi.sh \
    && PIXI_HOME=/opt/pixi PIXI_NO_PATH_UPDATE=1 PIXI_VERSION=${PIXI_VERSION} bash /tmp/install-pixi.sh \
    && rm -rf /var/lib/apt/lists/* /tmp/install-pixi.sh
ENV PATH="/opt/pixi/bin:${PATH}" \
    PIXI_NO_TELEMETRY=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace
COPY . .
CMD ["bash", "docker/smoke/check.sh"]
