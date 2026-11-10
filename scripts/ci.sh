#!/bin/sh
set -eu
docker build --file Dockerfile.prepare --tag local-semantic-atlas-prepared:0.1.0 .
docker build --network none --file Dockerfile --tag local-semantic-atlas:0.1.0 .
docker run --rm --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 256 --memory 1g --cpus 2 --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m local-semantic-atlas:0.1.0
