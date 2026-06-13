# SkinTokens Modly Extension

SkinTokens mesh auto-rigging extension for Modly. It wraps the upstream [VAST-AI SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) / TokenRig runtime and exposes a Modly `process` node that converts an input mesh into a rigged `.glb` workflow output.

This repository contains the Modly extension shell only: manifest, setup entrypoint, runtime adapter, readiness checks, and tests. SkinTokens model weights are not committed to this repository.

## What this extension provides

- **Install / Repair** through `setup.py` with JSONL progress events for the Modly UI.
- A Modly `process` node named **Rig Mesh with SkinTokens**.
- Deterministic SkinTokens source staging under the extension runtime directory.
- SkinTokens checkpoints and required config files stored under Modly's normal `models/` storage.
- Runtime progress stages for setup, readiness, SkinTokens inference, GLB export, and output validation.
- Fail-closed GPU compatibility checks for the current FlashAttention-based upstream runtime.

## Requirements

- Modly with GitHub/process extension support.
- Python `>= 3.11` provided by the Modly setup payload.
- NVIDIA CUDA GPU.
- **NVIDIA Ampere / RTX 30-series or newer is required** (`sm_80+`). RTX 20-series, GTX, and older GPUs are not supported because upstream SkinTokens uses FlashAttention in the TokenRig and SkinVAE paths.
- At least **14 GB VRAM** is recommended by the upstream runtime profile.
- CUDA `>= 12.1`; this extension prepares the PyTorch `cu128` lane by default.
- Internet access to download SkinTokens source/assets from GitHub and Hugging Face.
- Compatible `flash-attn`, `bpy`/Blender, and native Python dependencies for the target platform.

If the host GPU is older than Ampere and Modly passes GPU metadata, setup fails early with `gpu-too-old` before heavy installs/downloads. If GPU metadata is not available during setup, runtime performs the same check before generation.

## Install from Modly

1. In Modly, install this repository from GitHub:

   ```text
   https://github.com/DrHepa/modly-skintokens-extension
   ```

2. Let Modly run the extension setup/repair action.
3. Wait until setup reports the runtime as ready.
4. Load a supported mesh (`.obj`, `.fbx`, `.glb`, or `.gltf`) and run **Rig Mesh with SkinTokens**.

SkinTokens intentionally does **not** use Modly's generic node-level **Download** button. Setup owns the custom model layout because the runtime needs SkinTokens checkpoints, the Qwen config snapshot, and small upstream config resources in specific paths.

Do not validate Modly behavior by running setup from an arbitrary source checkout. The production path is Modly installing the extension into its `extensions/` directory and passing the setup payload.

## What setup does

Real setup prepares both the extension runtime and Modly model assets:

```text
extensions/skintokens-process-extension/
├── venv/
└── .skintokens-runtime/
    ├── bootstrap_state.json
    ├── vendor/skintokens/              # staged upstream SkinTokens source
    └── wheelhouse/flash-attn/          # optional local/managed flash-attn wheels

models/skintokens-process-extension/tokenrig/
├── configs/                            # synced upstream runtime configs
├── experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
├── experiments/skin_vae_2_10_32768/last.ckpt
└── models/Qwen3-0.6B/config.json       # Qwen config/non-weight files only
```

Setup actions:

- checks GPU capability when Modly provides `gpu_sm` / compute-capability metadata;
- creates an extension-owned virtual environment;
- installs `torch==2.7.0`, `torchvision==0.22.0`, and `torchaudio==2.7.0` from the `cu128` PyTorch lane;
- installs generic SkinTokens runtime dependencies;
- installs and probes required native packages such as `flash-attn` and `fast-simplification`;
- probes optional `open3d` availability for future voxel postprocess support;
- resolves the Blender bridge through either PyPI `bpy` or an external Blender executable;
- downloads/stages the upstream SkinTokens source snapshot;
- syncs required upstream runtime resources such as `configs/` into the model root;
- downloads and verifies Hugging Face sentinel assets;
- runs `pip check` and import probes;
- writes `.skintokens-runtime/bootstrap_state.json`.

On Windows `cp311/win_amd64`, setup can download and verify a managed `flash-attn` wheel for the PyTorch `2.7/cu128` lane. On platforms without a compatible `flash-attn` binary, build a local wheel into the extension wheelhouse and rerun setup:

```bash
python3 setup.py --build-flash-attn-wheel --max-build-jobs 2 '{"python_exe":"/usr/bin/python3","ext_dir":"/path/to/Modly/extensions/skintokens-process-extension"}'
python3 setup.py '{"python_exe":"/usr/bin/python3","ext_dir":"/path/to/Modly/extensions/skintokens-process-extension"}'
```

Normal setup then installs from:

```text
extensions/skintokens-process-extension/.skintokens-runtime/wheelhouse/flash-attn/
```

Source-building `flash-attn` during normal setup is disabled by default to avoid unbounded end-user installs. If you explicitly want the old source-build behavior, pass `--allow-flash-attn-source-build` or set `MODLY_SKINTOKENS_ALLOW_FLASH_ATTN_SOURCE_BUILD=1`.

## Runtime contract

- Extension id: `skintokens-process-extension`
- Node id: `rig-mesh`
- Node name: `Rig Mesh with SkinTokens`
- Input: mesh file path (`.obj`, `.fbx`, `.glb`, `.gltf`)
- Output: rigged `.glb`
- Canonical Modly output path: `workspaceDir/Workflows/<input-stem>_skintokens.glb`
- Fallback output path when `workspaceDir` is missing: extension runtime run directory

The processor reads one JSON object line on stdin and emits one JSON object per stdout line. Modly-visible event types are stable:

```jsonl
{"type":"progress","percent":5,"label":"Validating mesh input","stage":"validate-input"}
{"type":"log","message":"SkinTokens runtime assets are ready","stage":"readiness"}
{"type":"done","result":{"filePath":"/path/to/workspace/Workflows/model_skintokens.glb"}}
```

Errors are also JSONL:

```jsonl
{"type":"error","message":"SkinTokens requires an NVIDIA Ampere / RTX 30-series or newer GPU...","code":"gpu-too-old","stage":"gpu-preflight"}
```

Runtime progress/error stages:

1. `validate-input`
2. `gpu-preflight`
3. `readiness`
4. `bpy-server`
5. `load-model`
6. `prepare-mesh`
7. `encode-mesh`
8. `generate-tokens`
9. `decode-skin`
10. `postprocess`
11. `export-glb`
12. `validate-output`

Public parameters:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `top_k` | `5` | Top-k sampling for TokenRig autoregressive generation. |
| `top_p` | `0.95` | Nucleus sampling threshold. |
| `temperature` | `1.0` | Sampling temperature. |
| `repetition_penalty` | `2.0` | Penalty for repeated output tokens. |
| `num_beams` | `10` | Beam count used by upstream generation. |
| `use_skeleton` | `false` | When true, SkinTokens skins an input mesh that already includes a skeleton. |
| `use_transfer` | `true` | Uses upstream transfer export path to preserve texture and scale when possible. |

Voxel skin postprocess remains an upstream experimental path and is intentionally **not exposed** in the public Modly manifest until an `open3d`-compatible platform lane is validated. If a legacy/manual payload still sends `use_postprocess=true`, runtime checks `open3d` before loading models and fails early with `open3d-unavailable` when the dependency is missing.

## Support posture

This repository is public, but support claims remain evidence-based. A successful setup on one machine does not mean blanket support for every OS/GPU/Python combination.

| Host / capability | Status |
| --- | --- |
| NVIDIA Ampere / RTX 30-series or newer | Required for the current FlashAttention-based runtime. |
| RTX 20-series, GTX, or older NVIDIA GPUs | Unsupported; expected failure code is `gpu-too-old`. |
| Windows x86_64 / Python cp311 / PyTorch cu128 | Managed `flash-attn` wheel lane exists; real generation still requires a compatible Ampere+ CUDA host. |
| Linux ARM64 / Python cp312 / PyTorch cu128 | Maintainer development lane; may require a local `flash-attn` wheel in the extension wheelhouse. |
| Other hosts | Unvalidated until setup and real generation evidence exists. |

Keep these claims conservative. Do not mark a platform as supported until setup, model readiness, SkinTokens inference, GLB export, and Modly workspace retrieval have all been validated on that platform.

## Troubleshooting

- `gpu-too-old`: the GPU is below Ampere (`sm_80`). This is an upstream FlashAttention/runtime requirement, not a broken install. Use an RTX 30-series/Ampere-or-newer GPU.
- `cuda-unavailable`: PyTorch cannot see CUDA. Check NVIDIA driver, CUDA runtime, and the PyTorch lane installed in the extension venv.
- `flash-attn-wheel-unavailable`: no compatible binary wheel was found. Build a local wheel into `.skintokens-runtime/wheelhouse/flash-attn/` or use a validated platform lane.
- `bpy` install warning on Windows: setup tolerates the known PyPI metadata warning only when the import probe succeeds. If import fails, install/provide a compatible Blender executable.
- `open3d-unavailable`: a legacy/manual payload requested voxel skin postprocess, but `open3d` is not installed for this platform. The public Modly node no longer exposes this option. The base rigging path does not require voxel postprocess.
- Missing model/config files: rerun Modly setup/repair. SkinTokens assets are setup-owned and live under `models/skintokens-process-extension/tokenrig/`.
- No node-level **Download** button: expected. This extension does not use Modly's generic downloader because SkinTokens needs a custom asset layout.

## Development checks

Dry-run setup without installs/downloads:

```bash
python3 setup.py --dry-run
```

Run lightweight tests:

```bash
python3 -m unittest discover -s tests
```

The tests do not install dependencies, download model assets, import SkinTokens, import torch, or use a GPU.

## Upstream and licensing notes

This repository is a Modly integration wrapper around upstream SkinTokens. SkinTokens, TokenRig, PyTorch, Blender, Hugging Face-hosted assets, Qwen config files, and all third-party dependencies are governed by their own upstream licenses and terms.

Model weights are downloaded at setup time and are not redistributed in this repository.
