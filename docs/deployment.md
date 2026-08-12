# Deployment and hardware profiles

Panoptes is designed to run locally first and in a container second.

## Local requirements

| Component | Minimum for fixture mode | Recommended baseline |
|---|---:|---:|
| Python | 3.12 | 3.12 |
| Node.js | 20 | 22 LTS |
| RAM | 4 GB | 16 GB |
| Disk | 2 GB | 20 GB for baseline model caches |
| GPU | None | Optional CUDA GPU |

The baseline prose and code detectors should lazy-load. Do not load both unless the request requires both or the profile explicitly enables eager loading.

## Profiles

### `fixture`

Uses deterministic fixtures and no model downloads. This is the safest mode for CI, screenshots, UI development, and offline demos.

```bash
panoptes up --profile fixture
```

### `local-cpu`

Default desktop mode. Runs on CPU and downloads pinned artifacts into the user cache after verification.

```bash
panoptes up --profile local-cpu
panoptes doctor
```

### `local-gpu`

Optional accelerated mode. CUDA support is opt-in and must be reported by `panoptes doctor`. Metal acceleration is not promised unless a tested build explicitly enables it.

### `cloud-cpu` and `cloud-gpu`

Container profiles bind to `0.0.0.0:$PORT`. Local launchers bind to `127.0.0.1` by default.

## Local commands

```bash
panoptes up --profile fixture
panoptes doctor
panoptes analyze samples/prose-ai.txt
panoptes fixtures
panoptes models list
panoptes models verify
```

## Docker

```bash
docker build -t panoptes .
docker run --rm -p 8000:8000 -e PANOPTES_PROFILE=fixture panoptes
```

For baseline model execution, increase container memory and mount a cache volume only when the operator explicitly wants persistent model caches.

## Render deployment

`render.yaml` provisions one Linux web service. The service:

- builds the frontend;
- installs the backend package;
- serves static assets from FastAPI;
- binds `0.0.0.0:$PORT`;
- stores no submitted text by default.

A paid or appropriately sized instance is expected for real model inference. Use fixture mode only for demos on small instances.

## Operational checks

- `GET /healthz` returns process health.
- `GET /api/v1/runtime` reports profile, device, enabled detectors, artifact versions, and cache state.
- `GET /metrics` is disabled unless an operator token is configured.
