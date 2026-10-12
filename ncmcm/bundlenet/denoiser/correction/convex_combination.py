import torch
import torch.nn as nn
from ncmcm.bundlenet.denoiser.printer import print_status

from ncmcm.bundlenet.denoiser.denoiserlosses import LossRecorder

class ConvexCombinationCorrection(nn.Module):
    def __init__(self, input_dim: int, tolerance: float = 1e-6, bias: float = 0.0, tau=0.01, k=1.0, matching_weight=0.5, enforce_rank_constraint=False, bundlenet_tau=None):
        super(ConvexCombinationCorrection, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layer_size = int(self.input_dim * 1.5)
        self.tolerance = tolerance
        self.eps = 1e-6
        self.tau = tau
        self.k = k
        self.bias = bias
        self.inference = False
        self.bundlenet_tau = bundlenet_tau

        self.matching_weight = 1e-4
        self.derivative_weight = 1-self.matching_weight


        class LinearFullRank(nn.Module):
            def __init__(self, input_dim, output_dim, rank=3):
                super().__init__()
                self.linear_U = nn.Linear(input_dim, rank)
                self.linear_V = nn.Linear(rank, output_dim)

                nn.utils.parametrizations.orthogonal(
                    self.linear_U, 
                    "weight",
                    orthogonal_map="cayley"
                )
                nn.utils.parametrizations.orthogonal(
                    self.linear_V, 
                    "weight",
                    orthogonal_map="cayley"
                )

            def forward(self, x):
                return self.linear_V(self.linear_U(x))


        self.enforce_rank_constraint = enforce_rank_constraint
        self.module = None

        if self.enforce_rank_constraint:
            self.module = nn.Sequential(
                nn.Flatten(),
                LinearFullRank(self.input_dim * 2, self.hidden_layer_size, rank=3),
                LinearFullRank(self.hidden_layer_size, self.input_dim, rank=3)
            )
        else:
            self.module = nn.Sequential(
                nn.Flatten(),
                    nn.Linear(self.input_dim * 2, self.hidden_layer_size),
                nn.SiLU(),
                nn.Linear(self.hidden_layer_size, self.input_dim)
            )
        # limit the bias to be in the range [-1, 1]
        self.bias = max(-1.0, min(1.0, bias))

        self.loss_recorder = LossRecorder([])
        self.loss_recorder.register_loss("consistency_loss")
        self.loss_recorder.register_loss("matching_loss")
        
        print_status(message="Correction term initialized", details={
            "Input dim": self.input_dim,
            "Hidden layer size": self.hidden_layer_size,
            "τ": self.tau,
            "k": self.k,
            "β": self.bias
        })

    def record_training(self):
        self.loss_recorder.record_training()

    def record_testing(self):
        self.loss_recorder.record_testing()

    def name(self):
        return "ConvexCombinationCorrection"

    def set_inference(self):
        self.inference = True

    def set_bias(self, bias: float):
        """Permette di cambiare la temperatura anche dopo l'inizializzazione, a inferenza."""
        self.bias = bias

    def forward(self, xt, xt_previous, is_first):
        """
        alpha: peso sul DENOISED (f(x)). alpha=1 -> fidati completamente di f.
               alpha=0 -> fidati completamente del dato grezzo.
        """
        logits = self.module(torch.cat([xt, xt_previous], dim=1))
        alpha = torch.sigmoid(logits)

        alpha = alpha.reshape(xt.shape)

        alpha = (alpha + is_first).clamp(0, 1)  # Forza alpha=1 al primo passo, e alpha in [0,1] sempre

        return alpha 

    def apply_correction(self, denoised_xt, original_xt, original_xt_prev, is_first):
        alpha = self.forward(original_xt, original_xt_prev, is_first)

        # print(f"[bold cyan]Forward pass of ConvexCombinationCorrection. xt shape: {original_xt.shape}, xt_previous shape: {original_xt_prev.shape}, alpha shape: {alpha.shape}[/bold cyan]")    

        if self.inference:
            alpha = alpha + self.bias * (1-alpha) if self.bias > 0 else alpha * (1 + self.bias)
        corrected = alpha * denoised_xt  + (1 - alpha) * original_xt

        return corrected, alpha

    def loss(self, original_xt, original_xt_prev, f_xt, f_xt_prev, correction_t, correction_t_prev, is_first):
        """
        Calcola la loss di consistenza tra i valori corretti e i valori originali.

        f_xt, f_xt_prev: output del denoiser f, SENZA alcun mix con il grezzo
                          (serve isolato per calcolare il residuo).
        tau: soglia di distanza sopra la quale iniziare a preferire il dato grezzo.
             Se None, usa self.tolerance.
        """
        not_first = (1 - is_first)

        delta_obs = original_xt - original_xt_prev
        delta_f = f_xt - f_xt_prev
        distance = (delta_obs - delta_f).abs()

        # target: distanza alta -> alpha basso (fidati del grezzo)
        #         distanza bassa -> alpha alto (fidati di f)
        alpha_target = torch.sigmoid(-self.k * (distance - self.tau)).detach()

        # Compute the consistency loss
        diff_sq = (alpha_target - correction_t) ** 2
        consistency_loss = (diff_sq * not_first).sum() / (not_first.sum() * original_xt.shape[-1] + self.eps)

        corrected_t = f_xt * correction_t + original_xt * (1 - correction_t)
        matching_loss = ((corrected_t - original_xt).abs() * not_first).sum() / (not_first.sum() * original_xt.shape[-1] + self.eps)

        # print(f"[bold cyan]ConvexCombinationCorrection loss: {consistency_loss.item()}[/bold cyan]")

        self.loss_recorder.cache_loss("consistency_loss", consistency_loss.item())
        self.loss_recorder.cache_loss("matching_loss", matching_loss.item())

        return self.derivative_weight * consistency_loss + self.matching_weight * matching_loss

    def handle_epoch_end(self):
        self.loss_recorder.handle_epoch_end()

    def get_correction_loss_recordings(self):
        return self.loss_recorder.get_training_losses(), self.loss_recorder.get_testing_losses()
