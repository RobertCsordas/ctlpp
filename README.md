# Codebase for inspecting compatibility of representations on function interfaces

The official repository for our paper "CTL++: Evaluating Generalization on Never-Seen Compositional Patterns of Known Functions, and Compatibility of Neural Representations".

## Installation

This project requires Python 3 and PyTorch 1.8.

```bash
pip3 install -r requirements.txt
```

Create a Weights and Biases account and run 
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

For plotting, LaTeX is required (to avoid Type 3 fonts and to render symbols). Installation is OS specific.

### Running the experiments from the paper on a cluster

The code makes use of Weights and Biases for experiment tracking. In the ```sweeps``` directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run multiple configurations and seeds.

To reproduce our results, start a sweep for each of the YAML files in the ```sweeps``` directory. Run wandb agent for each of them in the _root directory of the project_. This will run all the experiments, and they will be displayed on the W&B dashboard. The name of the sweeps must match the name of the files in ```sweeps``` directory, except the ```.yaml``` ending. More details on how to run W&B sweeps can be found at https://docs.wandb.com/sweeps/quickstart. If you want to use a Linux cluster to run the experiments, you might find https://github.com/robertcsordas/cluster_tool useful.

### Running experiments locally

It is possible to run single experiments with Tensorboard without using Weights and Biases. This is intended to be used for debugging the code locally.
  
If you want to run experiments locally, you can use ```run.py```:

```bash
./run.py sweeps/fit_parallel_branches_2stage_overlap_ndr.yaml
```

If the sweep in question has multiple parameter choices, ```run.py``` will interactively prompt choices of each of them.

The experiment also starts a Tensorboard instance automatically on port 7000. If the port is already occupied, it will incrementally search for the next free port.

Note that the plotting scripts work only with Weights and Biases.

### Re-creating plots from the paper

Edit config file "paper/config.json". Enter your project name in the field "wandb_project" (e.g. "username/modules").

Run the script of interest within the "paper" directory. For example:

```bash
cd paper
python3 plot_1stage_performance.py
```

The output will be generated in the "paper/out/" directory.

# BibText
```
@article{csordas2022ctlpp,
      title={CTL++: Evaluating Generalization on Never-Seen Compositional Patterns of Known Functions, and Compatibility of Neural Representations}, 
      author={R\'obert Csord\'as and Kazuki Irie and J\"urgen Schmidhuber},
      booktitle={Proc. Conf. on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2022},
      month={December},
      address={Abu Dhabi, United Arab Emirates},
}
```
