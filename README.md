# CMPT 469 Final Project - Dynamic Radiant Foam

## Getting started

Start by cloning the repository and submodules:

    git clone --recursive https://github.com/quangminhdinh/CMPT469-Final

You will need a Linux environment with Python 3.10 or newer, as well as version 12.x of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and a CUDA-compatible GPU of Compute Capability 7.0 or higher. Please ensure that your installation method for CUDA places `nvcc` in your `PATH`. The following instructions were tested with Ubuntu 22.04.

After installing the CUDA Toolkit and initializing your python virtual environment, install PyTorch 2.3 or newer. For example, with CUDA 12.1:

    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

You might also have to install the following packages if you do not already have:

```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

From here, there are two options:

### Option 1: build with `pip install`

Choose this option if you want to run the code as-is, and do not need to make modifications to the CUDA/C++ code.

Simply run `pip install .` in the repository root. This will build the CUDA kernels and install them along with the python bindings into your python environment. This may take some time to complete, but once finished, you should be able to run the code without further setup.

Optionally if you want to install with the frozen version of required packages, you can do so by running `pip install -r requirements.txt` before running `pip install .`

### Option 2: build with CMake

Choose this option if you intend to modify the CUDA/C++ code. Using CMake directly will allow you to quickly recompile the kernels as needed.

First install the Python dependencies:

    pip install -r requirements.txt


Then, create a `build` directory in the repository root and run the following commands from it to initialize CMake and build the bindings library:

    cmake ..
    make install

This will install to a local `radfoam` directory in the repository root. Recompilation can be performed by re-running `make install` in the build directory.

### Data

Place the [D-NeRF](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0) and [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) datasets in `data/dnerf` and `data/N3V`.

To create a subset for the Neural 3D Video dataset to be used to train the model for the monocular scene reconstruction task, create a new scenario folder at `data/N3V/coffee_smal` and copy camera 0 and camera 7 from `data/N3V/coffee_martini` to it.

To use any scenario of the Neural 3D Video dataset, you need to extract the frames from the videos, downsample the frames, then run COLMAP to get the initial point positions.

First, to extract the frames from the videos of a scenario, run:

```shell
python scripts/extract.py data/N3V/$scene_name
```

The extracted images will be stored at `data/N3V/$scene_name/images`.

Next, if you just want to downsample the frame images, run:

```shell
pip install ImageMagic
bash scripts/resize.sh data/N3V/$scene_name
```

The x4 downsampled images will be stored at `data/N3V/$scene_name/images_4`. If you want to downsample the images to another factor, just change the corresponding configurations in `scripts/resize.sh`.

If you also want to run COLMAP to get the initial point positions, first follow the [official instructions](https://colmap.github.io/install.html) to install COLMAP from the source. Next, run:

```shell
bash scripts/local_colmap_and_resize.sh data/N3V/$scene_name
```

It will take some times to finish, but the results will be stored in `data/N3V/$scene_name/sparse/0`. This script will also downsample your frame images like `resize.sh`, so you just need to run one of the two.

## Train

To train the model for the multi-view scenario on the Neural 3D Video dataset, run:

    python dy_multi_train.py -c configs/n3v_multi.yaml

To train the model for the monocular scenario on the Neural 3D Video dataset, run:

    python dy_train.py -c configs/n3v.yaml

To train the model for the monocular scenario on the D-NeRF, run:

    python dy_train.py -c configs/dnerf.yaml

The test results will be stored in the experiment folder inside `dy_output`. Additionally, you can also observe the generated images during training in `dy_output/samp`.

If you encounter a problem with the training, you might have to downgrade your gcc and g++ version.

## Changelog

- Added the training codes for the monocular scenario (`dy_train.py`) and the multi-view scenario (`dy_multi_train.py`).
- Added different versions of the dynamic Radiant Foam models with temporal radial basis functions (`radfoam_model/dscene.py` and `radfoam_model/dscene_v1.py`).
- Added the SSIM and Ipips metrics (`metrics/`).
- Added the dataloaders for the D-NeRF dataset (`data_loader/blender.py`), the Neural 3D Video dataset for monocular scenario (`data_loader/n3v_simple.py`) and the Neural 3D Video dataset for multi-view scenario (`data_loader/n3v_multi.py`).
- Added the data preprocessing scripts: `scripts/resize.sh`, `scripts/local_colmap_and_resize.sh`, `scripts/extract.py`.

## Acknowledgments

This repository was implemented based on [Radiant Foam](https://github.com/theialab/radfoam)

    @article{govindarajan2025radfoam,
        author = {Govindarajan, Shrisudhan and Rebain, Daniel and Yi, Kwang Moo and Tagliasacchi, Andrea},
        title = {Radiant Foam: Real-Time Differentiable Ray Tracing},
        journal = {arXiv:2502.01157},
        year = {2025},
    }
