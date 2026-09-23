# Tasty Procedural Video: Features, Captioning and Weakly Supervised Step Segmentation

Research code from my internship at the **National University of Singapore** (Computer Vision & Machine Learning group, May–Jul 2019), working with Fadime Sener and Angela Yao on instructional-video understanding over the **Tasty Videos** dataset.

This work contributed to:

> **Transferring Knowledge from Text to Video: Zero-Shot Anticipation for Procedural Actions**
> Fadime Sener, Rishabh Saraf, Angela Yao — *IEEE TPAMI, 2022*
> [arXiv:2106.03158](https://arxiv.org/abs/2106.03158) · [DOI](https://doi.org/10.1109/TPAMI.2022.3218596)

> **Note.** This is the original research code, published as an archive rather than as a maintained library. It expects the Tasty Videos data on disk and carries hard-coded cluster paths from 2019 (see *Caveats*). It is here to show how the pipeline was built, not to run out of the box.

---

## The problem

A Tasty recipe video shows a sequence of steps ("chop the onions", "fold in the flour"), and each recipe comes with its written steps — but nobody has labelled *which frames* correspond to *which step*. Frame-level labels for thousands of videos are far too expensive to annotate.

So the pipeline below learns step segmentation from **weak supervision**: the ordered list of steps per video, without any frame-level timing.

```
video ──► frames ──► ResNet-50 / I3D features ─┐
                                               ├──► NN-Viterbi training ──► frame-level step segmentation
recipe steps (text) ──► embeddings ──► k-means ┘        (transcripts only, no frame labels)
                                    (200 classes)

frames + features ──► S2VT encoder-decoder ──► step captions
```

---

## What's in here

### `features/` — visual representations
- **`resnet50_features.py`** — ResNet-50 (ImageNet, final FC removed) run per frame to give a 2048-d vector, saved per recipe as one `resnet50.npy` of shape `(frames, 2048)`.
- **`i3d_features.py`** — I3D clip features (RGB stream), for motion-aware representations rather than per-frame appearance.
- **`videotransforms.py`** — crop/flip transforms for clip tensors, from [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d).

### `segmentation/` — weakly supervised step segmentation
- **`sentence_clustering.py`** — the interesting part. Recipe steps are free text, so there is no fixed label set. Step-sentence embeddings across all recipes are clustered with k-means into **200 pseudo-classes**, turning "whisk the eggs" and "beat the eggs together" into the same action label. Each recipe then becomes a sequence of cluster ids.
- **`create_transcript.py`** — turns each recipe into its ordered transcript of step classes, which is the only supervision the model sees.
- **`create_ground_truth.py`** — builds frame-level labels from the manual step/frame alignments, used **for evaluation only**.
- **`train.py`** — Neural-Network-Viterbi training: alternates between inferring a frame-to-step alignment with Viterbi and updating the frame classifier on that alignment.
- **`inference.py`** — Viterbi decoding of held-out videos, parallelised across processes.
- **`eval.py`** — frame accuracy against the ground-truth labels.
- **`utils/`** — the model pieces: `network.py` (frame classifier, trainer, forwarder), `viterbi.py` (decoder with hypothesis pruning), `grammar.py` (which step may follow which), `length_model.py` (Poisson step-duration model), `dataset.py` (feature/transcript loading).

### `captioning/` — S2VT step captioning
A PyTorch **S2VT** (sequence-to-sequence video-to-text) model that generates a caption per recipe step: `module.py` (two-layer LSTM encoder-decoder), `utils.py` (batching, vocabulary, frame sampling), plus `train.py`, `test.py` and `evaluate.py`.

### `splits/`
The recipe ids used for training and evaluation: **3,621 train / 399 val / 399 test**.

---

## Pipeline, end to end

1. **Frames** — decode each recipe video, resize to 40%, write JPEG frames.
2. **Features** — `features/resnet50_features.py` (per frame) and/or `features/i3d_features.py` (per clip).
3. **Label space** — `segmentation/sentence_clustering.py` clusters step-sentence embeddings into 200 classes.
4. **Weak labels** — `segmentation/create_transcript.py` writes each recipe's ordered class sequence.
5. **Train** — `segmentation/train.py` (NN-Viterbi, 10k iterations, lr 0.01 decayed 10× at 500).
6. **Decode and score** — `segmentation/inference.py`, then `segmentation/eval.py` for frame accuracy.
7. **Captions** — `captioning/train.py`, then `test.py` / `evaluate.py`.

---

## Caveats

- **The dataset isn't here.** Tasty videos, extracted frames and precomputed features are not redistributed, so nothing in this repo runs standalone. The paper above describes how the data was collected.
- **Paths come from environment variables**, defaulting to the 2019 cluster layout:
  `TASTY_DATA_ROOT` (recipe features and annotations), `TASTY_PROJECT_ROOT` (splits and cluster-label json), `S2VT_ROOT` (captioning checkpoints and logs).
  `captioning/utils.py` still has paths hard-coded inline.
- **`captioning/` expects a `data_process` module** (vocabulary building) from the upstream S2VT repo, which is not included here.
- **Model checkpoints are not included.**
- Written against Python 3.6 with PyTorch 1.x, TensorFlow 1.x (for I3D), OpenCV and scikit-learn.

## Known issues

Kept visible rather than quietly patched, since this is research code:

- `captioning/` cannot be imported as is: `module.py` and `utils.py` expect a `data_process` module (vocabulary building) that lived in the upstream S2VT repo.
- Magic numbers throughout: 201 classes in `segmentation/utils/`, `sample_size = 30800` and `n_steps = 80` in `captioning/train.py`, hard-coded row counts in the preprocessing loops.
- No unit tests; correctness was checked against the paper results at the time, not by a test suite.
- Two bugs found while publishing this repo have been fixed: the transcript writer wrote to a literal `"base_path"` filename, and `sentence_clustering.py` did not accumulate its per-recipe offset when slicing cluster labels.

## Credits and licensing

This repository contains my own code plus adaptations of two open-source projects, kept under their original licenses:

- **`segmentation/`** adapts [alexanderrichard/NeuralNetwork-Viterbi](https://github.com/alexanderrichard/NeuralNetwork-Viterbi) (MIT), from *NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning*, Richard et al., CVPR 2018.
- **`captioning/`** adapts [YiyongHuang/S2VT](https://github.com/YiyongHuang/S2VT), a PyTorch implementation of *Sequence to Sequence — Video to Text*, Venugopalan et al., ICCV 2015.
- **`features/videotransforms.py`** is from [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) (Apache-2.0).

See [THIRD_PARTY.md](THIRD_PARTY.md). Data-collection scripts written by other members of the group are deliberately not included.

My own contributions are released under the MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{sener2022transferring,
  title   = {Transferring Knowledge from Text to Video: Zero-Shot Anticipation for Procedural Actions},
  author  = {Sener, Fadime and Saraf, Rishabh and Yao, Angela},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2022},
  doi     = {10.1109/TPAMI.2022.3218596}
}
```
