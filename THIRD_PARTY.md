# Third-party code

This repository includes and adapts code from the projects below. Their licenses apply to
those parts, and the original copyright notices are reproduced here.

## NeuralNetwork-Viterbi (MIT) — `segmentation/`

https://github.com/alexanderrichard/NeuralNetwork-Viterbi

`segmentation/train.py`, `segmentation/inference.py`, `segmentation/eval.py` and
`segmentation/utils/*` are adapted from that repository, with changes to load Tasty recipe
features, use clustered step transcripts as the label space, and decode in parallel.

> Copyright (c) 2018 Alexander Richard
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify, merge,
> publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
> to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
> FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.

Paper: A. Richard, H. Kuehne, A. Iqbal, J. Gall. *NeuralNetwork-Viterbi: A Framework for
Weakly Supervised Video Learning*. CVPR 2018.

## S2VT, PyTorch implementation — `captioning/`

https://github.com/YiyongHuang/S2VT

`captioning/*` is adapted from that repository for the Tasty Videos dataset. Please check
the upstream repository for its current license terms before reusing this code.

Paper: S. Venugopalan et al. *Sequence to Sequence — Video to Text*. ICCV 2015.

## pytorch-i3d (Apache-2.0) — `features/videotransforms.py`

https://github.com/piergiaj/pytorch-i3d

Used unmodified for clip-level crop and flip transforms. Licensed under the Apache License,
Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0.

Paper: J. Carreira, A. Zisserman. *Quo Vadis, Action Recognition? A New Model and the
Kinetics Dataset*. CVPR 2017.
