# D3Net: A Unified Speaker-Listener Architecture for 3D Dense Captioning and Visual Grounding

<p align="center"><img src="docs/teaser.gif" width="600px"/></p>

## Introduction

Recent studies on dense captioning and visual grounding in 3D have achieved impressive results. Despite developments in both areas, the limited amount of available 3D vision-language data causes overfitting issues for 3D visual grounding and 3D dense captioning methods. Also, how to discriminatively describe objects in complex 3D environments is not fully studied yet. To address these challenges, we present D3Net, an end-to-end neural speaker-listener architecture that can detect, describe and discriminate. Our D3Net unifies dense captioning and visual grounding in 3D in a self-critical manner. This self-critical property of D3Net also introduces discriminability during object caption generation and enables semi-supervised training on ScanNet data with partially annotated descriptions. Our method outperforms SOTA methods in both tasks on the ScanRefer dataset, surpassing the SOTA 3D dense captioning method by a significant margin.

Please also check out the project website [here](https://daveredrum.github.io/D3Net/index.html).

For additional detail, please see the D3Net paper:  
"[D3Net: A Unified Speaker-Listener Architecture for 3D Dense Captioning and Visual Grounding](https://arxiv.org/abs/2112.01551)"  
by [Dave Zhenyu Chen](https://daveredrum.github.io/), [Qirui Wu](), [Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html) and [Angel X. Chang](https://angelxuanchang.github.io/)  
from [Technical University of Munich](https://www.tum.de/en/) and [Simon Fraser University](https://www.sfu.ca/).

## Setup

The code is tested on Ubuntu 18.04 LTS with PyTorch 1.9.1 CUDA 11.1 installed. Please follow the instructions on the [PyTorch official website](https://pytorch.org/) to set up PyTorch with correct version first.

Install the necessary packages listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```

### PointGroup-Minkowski

As we implement PointGroup by ourselves, it is required to install the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) first in order to run our code. Please see the [installation instructions](https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation) for more details.

__Before moving on to the next step, please don't forget to set the relevant project/data root path in `conf/path.yaml`.__

### Data preparation
1. Download the ScanRefer dataset and unzip it under `data/`. 
2. Download the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Pre-process ScanNet data. A folder named `split_data/` will be generated under `data/scannet/` after running the following command. Roughly 26GB free space is needed for this step:
```shell
cd data/scannet/
python prepare_scannet.py
```
5. (Optional) Pre-process the multiview features from ENet. 

    a. Download [the ENet pretrained weights (1.4MB)](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and decompress [the extracted ScanNet frames (~13GB)](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip).

    c. Change the data paths in `conf/path.yaml` marked with __TODO__ accordingly.

    d. Extract the ENet features:
    ```shell
    cd data/scannet/
    python compute_multiview_features.py
    ```

    e. Project ENet features from ScanNet frames to point clouds; you need ~36GB to store the generated HDF5 database:
    ```shell
    python project_multiview_features.py --maxpool
    ```

### Scan2CAD

As learning the object relative orientations in the relational graph requires CAD model alignment annotations in Scan2CAD, please refer to the [Scan2CAD official release](https://github.com/skanti/Scan2CAD#download-scan2cad-dataset-annotation-data) (you need ~8MB on your disk). Once the data is downloaded, extract the zip file under `data/` and change the path to Scan2CAD annotations (`SCAN2CAD`) in `conf/path.yaml` . As Scan2CAD doesn't cover all instances in ScanRefer, please download the [mapping file](http://kaldir.vc.in.tum.de/aligned_cad2inst_id.json) and place it under `SCAN2CAD`. Parsing the raw Scan2CAD annotations by the following command:

```shell
python scripts/Scan2CAD_to_ScanNet.py
```

__And don't forget to refer to [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric#pytorch-180181) to install the graph support.__

## Usage

For the stage-wise training, we need to prepare module weights stated as follows. If you would like to play around our [checkpoint](https://www.dropbox.com/s/nsrbcfeihmh2bhw/D3Net.7z?dl=0), please download and unzip it under `outputs`.

### Prepare PointGroup detector

Run the following script to start training the PointGroup detection backbone using the multiview features and normals:

```shell
python scripts/train.py --config conf/pointgroup.yaml
```

The trained model as well as the intermediate results will be dumped into `outputs/<output_folder>`. For evaluating the model (mAP@0.5), please run the following script - You should get mAP@0.5 around 47 at this point:

```shell
python scripts/eval.py --folder <output_folder> --task detection
```

Prepare the PointGroup weights for training the speaker:

```shell
python scripts/prepare_weights.py --path <path_to_checkpoint>  --config conf/pointgroup.yaml --model detector --model_name pointgroup
```

The prepared weights will be put under `pretrained`.

### Prepare Speaker

We will now fine-tune the PointGroup detector and train the speaker module with XLE loss:

```shell
python scripts/train.py --config conf/pointgroup-speaker.yaml
```

For evaluating the model (CIDEr@0.5IoU), please run the following script - You should get CIDEr@0.5IoU around 46 at this point:

```shell
python scripts/eval.py --folder <output_folder> --task captioning
```

> NOTE: We recommend compiling the `box_intersection.pyx` for faster evaluation:
> ```shell
> cd lib/utils && python cython_compile.py build_ext --inplace
> ```

Prepare the fine-tuned PointGroup weights and speaker checkpoint for next steps:

```shell
# prepare detector weights
python scripts/prepare_weights.py --path <path_to_checkpoint>  --config conf/pointgroup-speaker.yaml --model detector --model_name detector

# prepare speaker weights
python scripts/prepare_weights.py --path <path_to_checkpoint>  --config conf/pointgroup-speaker.yaml --model speaker --model_name speaker
```

### Prepare Listener

After the detector is fine-tuned, let's move on to the listener module:

```shell
python scripts/train.py --config conf/pointgroup-listener.yaml
```

For evaluating the model (Acc@0.5IoU), please run the following script - You should get Acc@0.5IoU around 35 at this point:

```shell
python scripts/eval.py --folder <output_folder> --task grounding
```

Prepare the listener checkpoint for next steps:

```shell
python scripts/prepare_weights.py --path <path_to_checkpoint>  --config conf/pointgroup-listener.yaml --model listener --model_name listener
```

### End-to-end joint training

Finally, it is time to put everything together for the joint speaker-listener training!

```shell
python scripts/train.py --config conf/pointgroup-speaker-listener.yaml
```

For evaluating the model performance, please run the following script - Note that since we're using reinforcement learning, you'll expect some variance in the trained model:

```shell
# detection
python scripts/eval.py --folder <output_folder> --task detection

# grounding
python scripts/eval.py --folder <output_folder> --task grounding

# captioning
python scripts/eval.py --folder <output_folder> --task captioning
```

## Citation
If you found our work helpful, please kindly cite the relavant papers:
```bibtex
@misc{chen2022d3net,
   title={D3Net: A Speaker-Listener Architecture for Semi-supervised Dense Captioning and Visual Grounding in RGB-D Scans}, 
   author={Dave Zhenyu Chen and Qirui Wu and Matthias Nießner and Angel X. Chang},
   year={2021},
   eprint={2112.01551},
   archivePrefix={arXiv},
   primaryClass={cs.CV}
}

@inproceedings{chen2021scan2cap,
   title={Scan2Cap: Context-aware Dense Captioning in RGB-D Scans},
   author={Chen, Zhenyu and Gholami, Ali and Nie{\ss}ner, Matthias and Chang, Angel X},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={3193--3203},
   year={2021}
}

@InProceedings{chen2020scanrefer,
   title={ScanRefer: 3D Object Localization in RGB-D Scans Using Natural Language},
   author={Chen, Zhenyu and Chang, Angel X. and Nie{\ss}ner, Matthias},
   editor={Vedaldi, Andrea and Bischof, Horst. and Brox, Thomas and Frahm, Jan-Michael},
   booktitle={Computer Vision -- ECCV 2020},
   publisher={Springer International Publishing},
   pages={202--221},
   year={2020},
   isbn={978-3-030-58565-5}
}
```

## License
D3Net is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2022 Dave Zhenyu Chen, Qirui Wu, Matthias Nießner, Angel X. Chang
