<p align="left">
    <!-- PyPi badge -->
    <a href="https://badge.fury.io/py/kplanes-nerfstudio"><img src="https://badge.fury.io/py/kplanes-nerfstudio.svg" alt="PyPI version"></a>
    <!-- License badge -->
    <a href="LICENSE.txt">
        <img alt="license" src="https://img.shields.io/badge/license-BSD-blue">
    </a>
</p>

# K-Planes nerfstudio integration

This repository provides code to integrate the [K-Planes model](https://sarafridov.github.io/K-Planes) into [nerfstudio](https://docs.nerf.studio/en/latest/index.html).

<div align='center'>
    <img src="https://sarafridov.github.io/K-Planes/assets/intro_figure.jpg" height="200px"/>
</div>


It provides an alternative way to use k-planes in addition to the [official repository](https://github.com/sarafridov/K-Planes), which allows access to nerfstudio's in-browser viewer and additional training capabilities.
Beware that some details about the training procedure differ from the official repository.


## Installation

1. [Install nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html). This is `pip install nerfstudio`, but there are a few dependencies (e.g. `torch`, `tinycudann`) which may require further steps, so make sure to check their installation guide!
2. Install the k-planes nerfstudio integration (this repository): `pip install kplanes-nerfstudio`

## Included Models

Two models are included here:
 - `kplanes` which is tuned for the Synthetic NeRF dataset (i.e. chair, drums, etc.)
 - `kplanes-dynamic` which is tuned to the [DNeRF dataset](https://www.albertpumarola.com/research/D-NeRF/index.html) (dynamic, monocular video).

:exclamation: PRs are welcome for configurations tuned to different datasets :exclamation:

You can run the static model by calling (remember to use the correct data directory!)
```
ns-train kplanes --data <data-folder>
```
and connect to the viewer using the link provided in the output of the training script.


## Benchmarks

**Synthetic NeRF** (hybrid model)

|      | drums  | materials | ficus  | ship   | mic    | chair  | lego  | hotdog | AVG    |
| ---- | -----  | --------- | -----  | ----   | ---    | -----  | ----  | ------ | ---    |
| PSNR | 26.31  | 29.82     | 32.47  | 30.27  | 33.73  | 34.98  | 36.56 | 36.77  | 32.61  |
| SSIM | 0.9394 | 0.9539    | 0.9788 | 0.8755 | 0.9857 | 0.9824 | 0.982 | 0.9792 | 0.9596 |


**DNeRF** (hybrid model)

|      | hell warrior | mutant | hook   | balls  | lego  | t-rex  | stand up | jumping jacks | AVG   |
| ---- | ------------ | ------ | ----   | -----  | ----  | -----  | -------- | ------------- | ---   |
| PSNR | 25.06        | 34.29  | 28.22  | 43.02  | 27.03 | 33.59  | 34.04    | 33.43         | 32.33 | 
| SSIM | 0.9487       | 0.9839 | 0.9552 | 0.9954 | 0.956 | 0.9817 | 0.9835   | 0.9797        | 0.973 |


## Roadmap

Expected future updates to this repository:

 - [ ] Including all datasets used in the K-Planes paper
 - [ ] Clarifying configuration of colliders (near-far)
 - [ ] Add benchmarks and configs for *linear* models


