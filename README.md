# CARN-MS

**CAD-Aware Rule-Gated and Network-Guided Mesh Simplification (CARN-MS)** is a
mesh simplification method that combines an area-weighted quadric error metric,
a topology and normal-inversion gatekeeper, and a neural network for ranking
candidate edge collapses. This repository provides the implementation and
pretrained weights for the neural priority model.

## Repository contents

| File | Description |
| --- | --- |
| `carn_ms.py` | Implementation and command-line interface for mesh simplification, quality evaluation, and development training. |
| `carn_ms_pretrained.pth` | Pretrained `EdgeDecisionNet` weights for neural-guided edge-collapse ranking. |

Input meshes and training datasets are not included. Provide your own triangle
meshes in OBJ format. The input should have consistent face winding, manifold
topology, and no degenerate or duplicate triangles.

## Installation

The CPU workflow has been verified on Linux with Python 3.12 and the package
versions below. A GPU is not required.

```bash
git clone https://github.com/frankzhaoyong/CARN-MS.git
cd CARN-MS
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.2.6 scipy==1.15.3 networkx==3.4.2 trimesh==4.6.13
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run these commands from the repository directory. Replace the example OBJ paths
with your own files.

### Simplify a mesh

```bash
python carn_ms.py --mode simplify \
  --input input.obj --output results/simplified.obj --ratio 0.5 \
  --checkpoint carn_ms_pretrained.pth --device cpu
```

`--ratio` specifies the target fraction of **vertices** to retain. The topology
gatekeeper may prevent reaching the requested target. The command exports the
simplified mesh and reports Chamfer distance (CD), curvature/normal error (CE),
feature preservation error (FDPE), and an auxiliary Hausdorff distance estimate.

When `--checkpoint` is omitted, simplification first checks
`./checkpoints/carn_ms_net.pth`, then uses the bundled `carn_ms_pretrained.pth`.
If neither is available, it runs the rule-only variant.

### Evaluate a simplified mesh

```bash
python carn_ms.py --mode evaluate \
  --input input.obj --output results/simplified.obj
```

Evaluation uses 5,000 surface samples across three seeds by default. For a
quicker check, add `--metric_samples 256 --metric_seed_count 2`.

### Train a new priority model

```bash
python carn_ms.py --mode train \
  --train_models mesh_a.obj mesh_b.obj \
  --checkpoint results/carn_ms_custom.pth --device cpu \
  --epochs 300 --patience 30
```

This development entry point generates edge-level training samples from the
supplied meshes and saves a new checkpoint. Use a separate output path to
preserve the bundled pretrained weights. The additional experiment scripts
mentioned in the source are not included; this repository does not provide the
complete training-data or paper-benchmark pipeline.

Use `python carn_ms.py --help` for all available options.

## Availability of Data and Materials

Suggested wording for the manuscript:

> The source code and pretrained model weights for CARN-MS are publicly
> available at https://github.com/frankzhaoyong/CARN-MS.
