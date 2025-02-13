# bridger_grasp

This repo contains code that's based on the following repos: [robotgradient/grasp_diffusion](https://github.com/robotgradient/grasp_diffusion).

## Environment Setup
Follow the steps in [robotgradient/grasp_diffusion](https://github.com/robotgradient/grasp_diffusion).

## Usage
To train BRIDGER, run the following:
```
python train.py --spec_file pcl_si --model_name si
```

You can modify the objects you want to train by setting 'class_type' in 'PointcloudAcronymAndSDFDataset()'.

To use data-drive source policy, train cvae first and then train BRIDGER

```
e.g. Please keep the seed and data_size to be the same
python train.py --spec_file pcl_cvae --model_name cvae
python train.py --spec_file pcl_si --model_name si
```

To generate grasp samples

```
e.g. python scripts/sample/generate_pcl_si.py --prior_type heuristic
```

To use the provided model checkpoint, you need to unzip the `pcl_heuristic.zip` file in the appropriate directory.

```commandline
bridger_grasp
...
data
└── models/
    └── pcl_heuristic/
        ├── model.pth  # (Example: Model checkpoint file)
        ├── params.json   # (Example: Metadata related to the model)

```

Then, run
```
python scripts/sample/generate_pcl_si.py --prior_type heuristic
```

## BibTeX

To cite this work, please use:

```
@article{chen2024behavioral,
  title={Don’t Start from Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion},
  author={Chen, Kaiqi and Lim, Eugene and Lin, Kelvin and Chen, Yiyang and Soh, Harold},
  journal={arXiv preprint arXiv:2402.16075},
  year={2024}
}
```