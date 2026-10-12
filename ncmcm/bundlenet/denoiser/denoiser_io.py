import torch
import torch.nn as nn

from ncmcm.bundlenet.denoiser.denoiser import Denoiser
from ncmcm.bundlenet.bundlenet import BunDLeNet

import os
from rich import print

import numpy as np
import yaml

class DenoiserIO:
    @classmethod
    def _create_config(cls, denoiser: Denoiser, bundlenet: BunDLeNet):
        denoiser_init_config = denoiser.get_init_config()

        number_of_neurons = denoiser_init_config.pop("number_of_neurons", None)
        _ = denoiser_init_config.pop("bundlenet_model", None)


        window = denoiser_init_config["window_size"]
        bundlenet_input_shape = (1, 1, window, number_of_neurons)

        bundlenet_init_config = {
            "latent_dim": bundlenet.latent_dim,
            "num_behaviour": bundlenet.num_behaviour,
            "input_shape": bundlenet_input_shape
        }
        
        config = {
            "bundlenet": bundlenet_init_config,
            "denoiser": denoiser_init_config
        }

       
        return config


    @classmethod
    def save(cls, denoiser: Denoiser, path: str):

        bundlenet : BunDLeNet = denoiser.bundlenet_model
        config = cls._create_config(denoiser, bundlenet)

        os.makedirs(path, exist_ok=True)
        torch.save(denoiser.state_dict(), os.path.join(path, "denoiser.pt"))
        torch.save(bundlenet.state_dict(), os.path.join(path, "bundlenet.pt"))

        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    @classmethod
    def load(cls, path: str, denoiser_class_type: type) -> tuple[Denoiser, BunDLeNet]:
        with open(os.path.join(path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundlenet_config = config["bundlenet"]
        denoiser_config = config["denoiser"]

        bundlenet = BunDLeNet(
            **bundlenet_config
        )

        denoiser = denoiser_class_type(
            bundlenet_model=bundlenet,
            **denoiser_config            
        )

        denoiser.load_state_dict(torch.load(os.path.join(path, "denoiser.pt"), map_location=device))
        bundlenet.load_state_dict(torch.load(os.path.join(path, "bundlenet.pt"), map_location=device))

        return denoiser, bundlenet

    @classmethod
    def save_indices(cls, train_indices, test_indices, path: str):
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "train_indices.npz"), indices=train_indices)
        np.savez(os.path.join(path, "test_indices.npz"), indices=test_indices)

    @classmethod
    def load_indices(cls, path: str):
        train_indices = np.load(os.path.join(path, "train_indices.npz"))
        test_indices = np.load(os.path.join(path, "test_indices.npz"))
        return train_indices['indices'], test_indices['indices']