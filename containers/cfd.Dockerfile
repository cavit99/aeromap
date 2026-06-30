FROM ubuntu:24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        wget \
        gnupg \
    && wget -O /etc/apt/trusted.gpg.d/openfoam.asc https://dl.openfoam.org/gpg.key \
    && add-apt-repository http://dl.openfoam.org/ubuntu \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        openfoam13 \
        python3 \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash foam

USER foam
WORKDIR /work

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["source /opt/openfoam13/etc/bashrc && foamRun -help"]
