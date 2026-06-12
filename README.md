# SkinTokens Process Extension for Modly

Process-extension wrapper for [VAST-AI SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), planned for Modly mesh-to-rigged-GLB workflows.

This repository intentionally starts from the Modly contract first:

- stdout from `processor.py` is JSONL-only for Modly UI consumption;
- install/setup emits observable JSONL events;
- runtime stages are mapped explicitly instead of hiding the upstream pipeline behind `demo.py`;
- model assets live under an extension-owned runtime directory;
- unsafe logical paths are rejected before setup/runtime use.

## Current status

This repository contains the Modly process wrapper, observable setup contract, and lightweight test harness. It does **not** claim a validated SkinTokens runtime lane yet; real platform support must be proven with setup + generation evidence.

Known upstream requirements from public SkinTokens evidence:

- Python >= 3.11
- NVIDIA GPU with at least 14 GB VRAM
- CUDA >= 12.1
- recommended PyTorch lane: `torch==2.7.0`, `torchvision==0.22.0`, `torchaudio==2.7.0` from `cu128`
- native/hard dependencies include `flash-attn`, `bpy>=4.2`, `open3d`, and `fast-simplification`
- public model repo: `VAST-AI/SkinTokens`

## Process protocol

`processor.py` reads one JSON object line from stdin and writes one JSON object per stdout line:

```jsonl
{"type":"progress","percent":5,"label":"Validating mesh input","stage":"validate-input"}
{"type":"log","message":"SkinTokens runtime assets are ready","stage":"readiness"}
{"type":"done","result":{"filePath":"/path/to/result.glb"}}
```

Errors are also JSONL:

```jsonl
{"type":"error","message":"rig-mesh requires input.filePath.","code":"protocol"}
```

## Runtime stages

The wrapper maps SkinTokens execution to these UI-visible stages:

1. `validate-input`
2. `readiness`
3. `bpy-server`
4. `load-model`
5. `prepare-mesh`
6. `encode-mesh`
7. `generate-tokens`
8. `decode-skin`
9. `postprocess`
10. `export-glb`
11. `validate-output`

The unit-test backend exercises this contract without importing SkinTokens, torch, bpy, or CUDA packages.

## Setup

`setup.py` supports two modes:

- `python3 setup.py --dry-run` — emits setup events and writes a dry-run readiness state without installing/downloading anything.
- Modly Install GitHub / `python3 setup.py '<payload-json>'` — performs real setup when Modly passes a payload such as `ext_dir` and `python_exe`.

Real setup prepares both the extension runtime and Modly model assets:

```text
extensions/skintokens-process-extension/
├── venv/
└── .skintokens-runtime/
    ├── bootstrap_state.json
    └── vendor/skintokens/          # upstream SkinTokens source snapshot

models/skintokens-process-extension/tokenrig/
├── experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
├── experiments/skin_vae_2_10_32768/last.ckpt
└── models/Qwen3-0.6B/config.json   # Qwen config/non-weight files only
```

Setup actions:

- create extension-owned venv;
- install exact torch/cu128 lane;
- install generic requirements;
- install flash-attn build prerequisites (`psutil`, `ninja`);
- install/probe required native/runtime packages such as `flash-attn` and `fast-simplification`;
  - first install `flash-attn` from `.skintokens-runtime/wheelhouse/flash-attn` if a local wheel exists;
  - otherwise try a binary-only pip install;
  - source-building `flash-attn` during normal setup is disabled by default to avoid unbounded user installs;
- try optional `open3d` installation for voxel postprocess support; if unavailable, base rigging can still proceed but `use_postprocess=true` will fail with an explicit runtime error;
- resolve the `bpy` provider:
  - first try `pip install bpy>=4.2` in the extension venv;
  - if that fails, probe a Blender executable from `blender_exe`, `MODLY_SKINTOKENS_BLENDER_EXE`, or `PATH` and record it as an `external-blender` fallback;
- download the upstream SkinTokens source snapshot;
- sync small upstream runtime resources such as `configs/` into the Modly model root because upstream resolves them relative to the runtime cwd;
- download HF sentinels from `VAST-AI/SkinTokens` into Modly `models/skintokens-process-extension/tokenrig`;
- download Qwen3-0.6B config only;
- run `pip check` and import probes;
- write `.skintokens-runtime/bootstrap_state.json`.

The process node intentionally does **not** declare node-level `hf_repo`. SkinTokens uses a custom setup-owned model layout (`models/skintokens-process-extension/tokenrig`) and also needs the Qwen config snapshot, so Modly's generic node download button would put files in the wrong owner path and would not represent setup readiness. Public model sources remain documented in manifest metadata and `asset_requirements` instead.

The runtime then runs upstream SkinTokens with `cwd` set to the Modly model root and `PYTHONPATH` pointing at the vendored source snapshot. This preserves upstream relative paths while keeping model assets in Modly's model storage.

When setup records `bpy_provider.kind = "external-blender"`, runtime starts `bpy_server.py` with:

```text
blender --background --factory-startup --python bpy_server.py
```

and extends `PYTHONPATH` with the vendored SkinTokens source plus extension venv site-packages. This is a fallback path for platforms where PyPI does not provide a compatible `bpy` wheel.

### Local flash-attn wheel flow

On platforms where `flash-attn` has no compatible binary wheel, build it once into the extension-local wheelhouse and then rerun setup:

```bash
python3 setup.py --build-flash-attn-wheel --max-build-jobs 2 '{"python_exe":"/usr/bin/python3","ext_dir":"/home/drhepa/Documentos/Modly/extensions/skintokens-process-extension"}'
python3 setup.py '{"python_exe":"/usr/bin/python3","ext_dir":"/home/drhepa/Documentos/Modly/extensions/skintokens-process-extension"}'
```

The wheel build writes to:

```text
extensions/skintokens-process-extension/.skintokens-runtime/wheelhouse/flash-attn/
```

Normal setup then installs from that local wheelhouse using `--no-index --find-links`. If you explicitly want the old source-build behavior during setup, pass `--allow-flash-attn-source-build` or set `MODLY_SKINTOKENS_ALLOW_FLASH_ATTN_SOURCE_BUILD=1`, but this is not recommended for end-user installs.

## Future Modly rigged-asset service

SkinTokens' upstream `bpy_server.py` is mostly a mesh/rig bridge:

- `/load` imports OBJ/FBX/GLB and extracts vertices, faces, armatures, joints, transforms, skin weights, and optional animation matrices into a SkinTokens `Asset`.
- `/export` converts a SkinTokens `Asset` into meshes, armature, vertex groups/skin weights, and GLB/FBX output.
- `/transfer` reloads the original mesh, estimates a similarity transform, transfers predicted skeleton/skin to the original geometry, then exports.

Long term, Modly should own this as a reusable rigged-asset capability instead of every extension carrying a Blender bridge:

```text
rigged-mesh/import
rigged-mesh/export
rigged-mesh/transfer-skin
```

That future service would benefit SkinTokens, UniRig, animation/retargeting workflows, and any tool that needs rigged GLB output without depending on `bpy_server.py`.

## Tests

Run lightweight tests only:

```bash
python -m unittest discover -s tests
```

The tests do not install dependencies, download model assets, import torch, import SkinTokens, or use GPU.
