# Mask R-CNN of TensorFlow 2

## Environment

Install TensorFlow 2.0 and other dependencies:

```bash
# CUDA 11.8 + cuDNN 8.6 + Python 3.10
conda env create -f conda-cu11-py310.yaml
# Apple Silicon + Python 3.11
conda env create -f conda-apple-py311.yaml
```

Uninstall:

```bash
conda env remove --name py311-apple-tfrcnn
```

Note that environment variables such as `LD_LIBRARY_PATH` must be set properly
on a GPU machine:

```fish
set -Ux CUDNN_PATH $CONDA_PREFIX/lib/python3.1/site-packages/nvidia/cudnn
set -Ux LD_LIBRARY_PATH $LD_LIBRARY_PATH $CONDA_PREFIX/lib $CUDNN_PATH/lib
set -Ux XLA_FLAGS --xla_gpu_cuda_data_dir=$CONDA_PREFIX
```

## Usage

Train the RPN part:

```bash
tf-rcnn train-rpn --epochs=5 --save-intv=10 --batch=32
```

Predict the region of interest (ROI):

```bash
tf-rcnn predict-rpn --images=20
```

Result of the ROI prediction after 5 epochs:

![ROI prediction](img/voc_2007_test_rpn_0017.jpg)

## References

- [Setup tensorflow 2.12 on casper](https://github.com/NCAR/casper_tensorflow_gpu)
