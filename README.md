# Flaxv3

Flaxv3 is a simplified version of DreamerV3, composed of Flax and JAX instead of Ninjax, which was used by Hafner in DreamerV3.

## Setup

To set up your Python environment, please install the required dependencies listed in `requirements.txt`:
```
pip install -r requirements.txt
```

## Configuration

To modify the hyperparameters for the algorithm, refer to the [config file](./config/config.yaml).

## Training

To train an agent, use the following command:
```
python train.py training.env_name=[env_name] training.seed=[seed] training.device=[available_device]
```

Make sure to replace `[env_name]` with Atari games like `Boxing`. The log files and checkpoints will be saved in the `outputs` directory.

## Evaluation

To evaluate the trained agent, run:
```
python eval.py eval.env_name=[env_name] eval.ckpt_path=[checkpoints] eval.device=[available_device]
```

Make sure to replace `[checkpoints]` with the path to the checkpoint file.
