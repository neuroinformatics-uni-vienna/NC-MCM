from ncmcm.bundlenet.denoiser.denoiser import Denoiser
import torch
import torch.nn as nn

from ncmcm.bundlenet.denoiser.printer import print_status, print_success

class Activation(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.nn.functional.elu(input=x,alpha=self.alpha) + self.alpha


class InjectiveLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=3):
        super().__init__()

        self.in_features = input_dim
        self.out_features = output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        self.linear_U = nn.Linear(input_dim, output_dim, bias=True)

        # enforce that weights are optimized without loss of rank
        nn.utils.parametrizations.orthogonal(
            self.linear_U, 
            "weight",
            orthogonal_map="cayley"
        )

        self.activation = Activation(alpha=1.0)

    def forward(self, x):
        x = self.linear_U(x)
        # x = self.linear_V(x)
        x = self.activation(x)
        return x

class FinalInjectiveLayer(InjectiveLayer):
    def __init__(self, input_dim, output_dim, rank=3):
        super().__init__(input_dim, output_dim, rank=rank)

        self.activation = nn.Softplus()

class InjectiveDenoiser(Denoiser):
    def __init__(self, 
        bundlenet_model, 
        window_size,
        den_hidden_layers=None,
        bundlenet_hidden_layers=None,
        add_correction=False,
        correction_bias=0.0,
    ):

        Denoiser.__init__(
            self,
            bundlenet_model=bundlenet_model,
            window_size=window_size,
            den_hidden_layers=den_hidden_layers,
            bundlenet_hidden_layers=bundlenet_hidden_layers,
            add_correction=add_correction,
            correction_bias=correction_bias,
        )

        # Here we basically only override the tau_d_inv function to ensure that it is structurally
        # injective.
        
        print_status(message="Injective denoiser overriding inverse function.", details={})


        self.tau_d_inv_hidden_layers = []
        if den_hidden_layers is None:
            # Layers: latent -> 10 -> 30 -> 50 -> input_dim
            self.tau_d_inv_hidden_layers = [
                InjectiveLayer(self.bundlenet_latent_dim, 10, rank=10),
                InjectiveLayer(10, 30, rank=30),
                InjectiveLayer(30, 50, rank=50),
                InjectiveLayer(50, self.bundlenet_input_features, rank=self.bundlenet_input_features),
            ]
        else:
            n_hidden_layers = len(den_hidden_layers)
            for i in range(n_hidden_layers -1):
                self.tau_d_inv_hidden_layers.append(
                    InjectiveLayer(den_hidden_layers[i], den_hidden_layers[i+1])
                )
        self.tau_d_inv = nn.Sequential(
            nn.Flatten(),
            *self.tau_d_inv_hidden_layers,
        )
