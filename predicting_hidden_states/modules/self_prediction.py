from functools import reduce
from typing import Callable, Optional, Union, Dict, Any, List

import numpy as np

import torch as th
import torch.nn
import torch.nn.functional as F
from torch import nn

from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.kv_cache import KVCache
from torchtune.modules.transformer import _get_clones

from torch import einsum
from scipy.cluster.vq import kmeans2
import wandb
from einops import repeat


def gaussian_kl(mu_q, log_var_q, mu_p, log_var_p):
    """
    Calculates the KL divergence between two diagonal Gaussian distributions.

    This function computes the Kullback-Leibler divergence $D_{KL}(q||p)$ where
    q and p are Gaussian distributions with diagonal covariance matrices.

    Args:
        mu_q (torch.Tensor): The mean of the first Gaussian distribution (q).
        log_var_q (torch.Tensor): The log-variance of the first Gaussian distribution (q).
        mu_p (torch.Tensor): The mean of the second Gaussian distribution (p).
        log_var_p (torch.Tensor): The log-variance of the second Gaussian distribution (p).

    Returns:
        torch.Tensor: A tensor containing the element-wise KL divergence.
    """
    kl = (log_var_q - log_var_p) + 0.5 * (
        torch.exp(2 * (log_var_p - log_var_q))
        + (mu_p - mu_q) ** 2 / torch.exp(log_var_p)
        - 1
    )
    return kl


class PHiMLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) with SwiGLU and residual connections.

    This module is a building block for the PHi layer, suitable for use in the
    prior predictor or decoder. Its architecture is determined
    by the `num_layers` parameter:
    - `num_layers = 1`: The MLP is a simple linear transformation.
    - `num_layers = 2`: The MLP uses a standard SwiGLU (Swish-Gated Linear Unit) block.
    - `num_layers > 2`: The MLP becomes a deep network of SwiGLU blocks with
      residual skip connections between them.

    Args:
        input_dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output features.
        num_layers (int): The number of layers, which dictates the architecture.
        activation (nn.Module, optional): The activation function to use within the
            SwiGLU blocks. Defaults to nn.SiLU().
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation: nn.Module = nn.SiLU()):
        super().__init__()
        if num_layers == 1:
            hidden_dim = input_dim
        self.gate_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        current_input_dim = input_dim
        for l in range(1, num_layers):
            self.gate_layers.append(nn.Linear(current_input_dim, hidden_dim))
            self.projection_layers.append(nn.Linear(current_input_dim, hidden_dim))
            current_input_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        residual = 0
        for gate_layer, projection_layer in zip(self.gate_layers, self.projection_layers):
            gate = self.activation(gate_layer(x))
            proj = projection_layer(x)
            x = residual + gate * proj
            residual = x
        return self.output_layer(x)


class PHiLayer(torch.nn.Module):
    """
    Implements the PHi (Prediction of Hidden states) layer.

    This module measures the complexity of a sequence model's computation
    by creating an information bottleneck on its hidden states and calculating
    a loss based on the model's ability to predict its own future states.

    Args:
        d_model (int): The dimensionality of the hidden states.
        posterior_mlp (torch.nn.Module): The network that encodes the hidden state
            into the parameters of the posterior distribution `q`.
        decoder_mlp (torch.nn.Module): The network that decodes the latent variable `z`
            back into a hidden state representation.
        prior_prediction_mlp (torch.nn.Module): The MLP part of the autoregressive
            prior model `p`.
        prior_prediction_attention (Optional[torch.nn.Module], optional): The attention
            part of the autoregressive prior model `p`. Defaults to None.
        sa_norm (Optional[nn.Module], optional): Normalization layer applied before the
            prior prediction MLP. Defaults to nn.Identity().
        self_critic_loss_factor (float, optional): Weighting factor for the self-critic
            loss, used to prevent posterior collapse.  Defaults to 0.1.
        next_loss_factor (float, optional): Weighting factor for the PHi loss. Defaults to 0.1.
        detach_hidden_states (bool, optional): If True, detaches the incoming hidden
            states from the computation graph. Defaults to False.
        detach_targets (bool, optional): If True, detaches the target distributions
            (the posterior) during PHi loss calculation. Defaults to False.
        full_information_blockage (bool, optional): If True, forces the latent
            variable `z` to have zero information, for ablation. Defaults to False.
        chance_to_deterministic (float, optional): Probability of making the sampling
            of `z` deterministic during training. Defaults to 0.0.
        deterministic_at_inference (bool, optional): If True, uses the mean of the
            posterior instead of sampling during inference. Defaults to False.
        straight_through_eval (bool, optional): If True, passes the original hidden
            state `h` through the layer during evaluation, bypassing the bottleneck.
            Defaults to False.
        use_information_bottleneck (bool, optional): If True, enables the variational
            information bottleneck. Defaults to True.
        use_hidden_state_prediction (bool, optional): If True, enables the self-prediction
            mechanism. Defaults to True.
    """
    def __init__(
        self,
        d_model: int,
        posterior_mlp: torch.nn.Module,
        quantizer_mlp: torch.nn.Module,
        decoder_mlp: torch.nn.Module,
        prior_prediction_mlp: torch.nn.Module,
        prior_prediction_attention: torch.nn.Module | List[nn.Module] | None = None,
        sa_norm: Optional[nn.Module] = None,
        self_critic_loss_factor: float = 0.1,
        next_loss_factor: float = 0.1,
        detach_hidden_states: bool = False,
        detach_targets: bool = False,
        full_information_blockage: bool = False,
        chance_to_deterministic: float = 0.0,
        deterministic_at_inference: bool = False,
        straight_through_eval: bool = False,
        use_information_bottleneck: bool = True,
        use_hidden_state_prediction: bool = True,
    ):
        super().__init__()
        self.posterior_mlp = posterior_mlp
        self.quantizer = quantizer_mlp
        self.decoder_mlp = decoder_mlp
        self.prior_prediction_mlp = prior_prediction_mlp
        self.prior_prediction_attention = prior_prediction_attention
        self.sa_norm = sa_norm or nn.Identity()
        self.next_loss_factor = next_loss_factor
        self.self_critic_loss_factor = self_critic_loss_factor
        self.initial_embedding = torch.nn.Parameter(torch.zeros(1, 1, d_model))

        self.detach_hidden_states = detach_hidden_states
        self.detach_targets = detach_targets
        self.full_information_blockage = full_information_blockage
        self.chance_to_deterministic = chance_to_deterministic
        self.deterministic_at_inference = deterministic_at_inference
        self.straight_through_eval = straight_through_eval
        self.use_information_bottleneck = use_information_bottleneck
        self.use_hidden_state_prediction = use_hidden_state_prediction

    def forward(self,
                h: torch.Tensor,
                padding_mask: torch.Tensor,
                mask: Optional[_MaskType] = None,
                input_pos: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Defines the forward pass of the PHi layer.

        Args:
            h (torch.Tensor): The input hidden states from the main model.
            padding_mask (torch.Tensor): The padding mask for the sequence.
            mask (Optional[_MaskType], optional): The causal attention mask for the
                autoregressive prior. Defaults to None.
            input_pos (Optional[torch.Tensor], optional): Positional encodings for
                the attention mechanism. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the new hidden state `h`,
                the PHi loss `phi_loss`, and other metrics.
        """
        return_dict = {}
        if self.detach_hidden_states:
            h = h.detach()
        padding_mask = ~padding_mask

        if self.quantizer is None:
            # --- Information Bottleneck ---
            # 1. Compute posterior distribution q(z|h) and sample latent z
            distribution = self.posterior_mlp(h)
            q_mean, q_logvar = distribution.chunk(2, dim=-1)
            q_logvar = torch.clamp(q_logvar, -5, 10)

            if self.full_information_blockage:
                # block all information in the latent space by having zero mean and log variance
                q_mean = q_mean * 0.0
                q_logvar = q_logvar * 0.0

            # 2. Sample from the posterior using the reparameterization trick
            use_information_bottleneck = self.training or not self.deterministic_at_inference
            use_information_bottleneck = use_information_bottleneck and self.use_information_bottleneck
            if use_information_bottleneck: # Only at training time
                noise = torch.exp(0.5 * q_logvar) * torch.randn_like(q_mean)
                if self.chance_to_deterministic > 0.0:
                    deterministic = torch.rand_like(q_mean[:, 0, 0]) < self.chance_to_deterministic
                    noise = noise * ~deterministic.view(-1, 1, 1)
                z = q_mean + noise
            else:
                z = q_mean

            # 3. Self-critic loss to prevent posterior collapse
            self_critic_scores = -F.gaussian_nll_loss(
                q_mean.unsqueeze(1),
                z.unsqueeze(0),
                q_logvar.exp().unsqueeze(1),
                reduction="none").sum(-1).transpose(1, 2)  # shape: (batch_size, seq_len, d_model)
            self_critic_targets = torch.arange(self_critic_scores.shape[2])[:, None].repeat(1, self_critic_scores.shape[1])
            self_critic_losses = F.cross_entropy(
                self_critic_scores.reshape(-1, self_critic_scores.shape[-1]),
                self_critic_targets.flatten().to(h.device),
                reduction="none",
            )
            self_critic_losses = (self_critic_losses * padding_mask.flatten()).view_as(padding_mask)
            self_critic_loss = self_critic_losses.sum() / padding_mask.sum()
            return_dict["self_critic_loss"] = self_critic_loss * self.self_critic_loss_factor

            # --- Self Prediction ---
            # 4. Compute autoregressive prior p(z_t | z_{<t})
            prediction_input = z
            if self.prior_prediction_attention is not None:
                prediction_input = self.prior_prediction_attention(prediction_input, prediction_input,
                                                                mask=mask, input_pos=input_pos)
            # Shift input for next-step prediction
            prediction_input = prediction_input[:, :-1]  # (batch_size, seq_len - 1, d_model)
            prediction_input = torch.cat((self.initial_embedding.expand(prediction_input.shape[0], -1, -1),
                                        prediction_input), dim=1)  # (batch_size, seq_len, d_model)
            prediction_mean = self.prior_prediction_mlp(self.sa_norm(prediction_input))

            if prediction_mean.shape[-1] == 2 * h.shape[-1]:
                # Split the prediction mean and log variance
                prediction_mean, prediction_logvar = prediction_mean.chunk(2, dim=-1)
                prediction_logvar = torch.clamp(prediction_logvar, -5, 10)

            if not self.use_hidden_state_prediction:
                # Use unit gaussian as a prior if hidden state prediction is not used
                prediction_mean = torch.zeros_like(prediction_mean)
                prediction_logvar = torch.zeros_like(prediction_logvar)

            # 5. Calculate PHi Loss (KL divergence between prior and posterior)
            target_mean = q_mean
            target_logvar = q_logvar
            if self.detach_targets:
                target_mean = target_mean.detach()
                target_logvar = target_logvar.detach()

            target_padding_mask = padding_mask

            if self.use_information_bottleneck:
                phi_losses = gaussian_kl(
                    mu_q=prediction_mean,
                    log_var_q=prediction_logvar,
                    mu_p=target_mean,
                    log_var_p=target_logvar,
                )
            else:
                phi_losses = F.mse_loss(prediction_mean, target_mean, reduction="none")
            phi_losses = phi_losses.mean(dim=-1) * target_padding_mask
            return_dict["tokenwise_phi_losses"] = phi_losses
            loss = phi_losses.sum() / target_padding_mask.sum()
            return_dict["phi_loss"] = loss * self.next_loss_factor

            if self.decoder_mlp is not None:
                h_new = self.decoder_mlp(z)
            # else:
            #     h_new = z

            # if self.straight_through_eval and not self.training:
            #     h_new = h

            return_dict["h"] = h_new
            return return_dict

        else:
            # z = self.posterior_mlp(h)
            h_new, latent_losses, inds, zs = self.quantizer(h)

            for i in range(self.quantizer.num_quantizers):
                wandb.log({"num unique idxs": len(set(inds[i].flatten().tolist()))})

            if self.log_hist:
                for i in range(self.quantizer.num_quantizers):
                    ind_hist, ind_bin_edges = np.histogram(inds[i].flatten().cpu().numpy(), bins=512, density=True)
                    wandb.log({f"custom/ind hist{i}": wandb.Histogram(np_histogram=(ind_hist, ind_bin_edges))})

            # Loss term -> KL divergence of the uniform prior / do not train but good to observe
            #TODO: rewrite to handle multiple quantizers
            latent_padding_mask = repeat(padding_mask, 'b s -> q b s', q=latent_losses.size(0)) if latent_losses.dim() == 4 else padding_mask
            latent_losses = latent_losses.sum(-1)
            latent_loss = latent_losses * latent_padding_mask
            for i in range(self.quantizer.num_quantizers):
                return_dict[f"tokenwise_latent_loss{i}"] = latent_loss[i]

            ### Self Critic Losses ###
            # TODO: rewrite to handle multiple quantizers

            for i in range(self.quantizer.num_quantizers):
                ind_one_hot = F.one_hot(inds[i], num_classes=zs[i].size(-1))  # (batch, seq_len, num_codes)
                self_critic_scores = torch.einsum('i s c, j s c -> i j s', zs[i], ind_one_hot.to(zs[i].dtype))  # (batch_z, batch_ind_one_hot, seq_len)
                self_critic_scores = self_critic_scores.transpose(1, 2)  # (batch_z, seq_len, batch_ind_one_hot)
                self_critic_targets = torch.arange(self_critic_scores.shape[2])[:, None].repeat(1, self_critic_scores.shape[1])  # (batch, seq_len)
                self_critic_losses = F.cross_entropy(
                    self_critic_scores.reshape(-1, self_critic_scores.shape[-1]).float(),
                    self_critic_targets.flatten().to(zs.device),
                    reduction="none",
                )
                self_critic_losses = (self_critic_losses * padding_mask.flatten()).view_as(padding_mask)
                return_dict[f"tokenwise_self_critic_loss{i}"] = self_critic_losses.reshape(-1, padding_mask.shape[1])
                self_critic_loss = self_critic_losses.sum() / padding_mask.sum()
                return_dict[f"self_critic_loss{i}"] = self_critic_loss * self.self_critic_loss_factor

            ### Self Prediction ################################################################
            # compute autoregressive prior based on the previous latent variables
            # TODO: the prediction has been modified to include additional term / CAREFUL for experiments
            # prediction_input = z_q
            # if self.prior_prediction_attention is not None:
            #     prediction_input = self.prior_prediction_attention(prediction_input, prediction_input,
            #                                                         mask=mask, input_pos=input_pos)
            # prediction_input = prediction_input[:, :-1]
            # prediction_input = torch.cat((self.initial_embedding.expand(prediction_input.shape[0], -1, -1),
            #                           prediction_input), dim=1)
            # prediction_z = self.prior_prediction_mlp(self.sa_norm(prediction_input))

            prediction_z = []
            prediction_input = self.quantizer.get_quantized_inputs # size: q, bsz, seq_len, dim
            if self.prior_prediction_attention is not None:
                for i, prior_attn_layer in enumerate(self.prior_prediction_attention):
                    prediction_input_post_attn = prior_attn_layer(prediction_input[i], prediction_input[i],
                                                                    mask=mask, input_pos=input_pos)
                    prediction_shift = prediction_input_post_attn[:, :-1]
                    prediction_concat_parameter = torch.cat((self.initial_embedding.expand(prediction_input[i].shape[0], -1, -1),
                                      prediction_shift), dim=1)
                    prediction_z.append(self.prior_prediction_mlp(self.sa_norm[i](prediction_concat_parameter)))
            
            prediction_z = torch.stack(prediction_z) # [q, bsz, seq_len, num_embeddings]

            # Calculate PHi loss (KL divergence between prior(input) and posterior(target))
            target_z = zs
            if self.detach_targets:
                target_z = target_z.detach()
            
            target_padding_mask = padding_mask

            categroical_input = F.log_softmax(prediction_z, dim=-1) # prior / log probs
            categorical_target = F.log_softmax(target_z, dim=-1) # posterior / log probs
            phi_losses = F.kl_div(categroical_input, categorical_target, reduction='none', log_target=True)   
            
            # # entropy of targets
            # target_entropy = (-categorical_target.exp() * categorical_target).sum(dim=-1)
            # input_entropy = (-categroical_input.exp() * categroical_input).sum(dim=-1)
            # return_dict["tokenwise_phi_target_entropy"] = target_entropy
            # return_dict["tokenwise_phi_input_entropy"] = input_entropy
            
            phi_losses = phi_losses.sum(dim=-1) * target_padding_mask
            for i in range(self.quantizer.num_quantizers):
                return_dict[f'tokenwise_phi_losses{i}'] = phi_losses[i]
            loss = phi_losses.sum() / target_padding_mask.sum()
            return_dict['phi_loss'] = loss * self.next_loss_factor
            # Two losses: 1. trains only the prior 2. trains only the posterior
            
            # 1. detach posterior ( train only the prior )
            # target_z = z
            # target_z = target_z.detach()
            
            # target_padding_mask = padding_mask

            # categroical_input = F.log_softmax(prediction_z, dim=-1) # prior / log probs
            # categorical_target = F.log_softmax(target_z, dim=-1) # posterior / log probs
            # phi_losses_prior = F.kl_div(categroical_input, categorical_target, reduction='none', log_target=True)
            
            # phi_losses_prior = phi_losses_prior.sum(dim=-1) * target_padding_mask
            # return_dict['tokenwise_phi_losses_prior'] = phi_losses_prior 
            # loss = phi_losses_prior.sum() / target_padding_mask.sum()
            # return_dict['phi_loss_prior'] = loss * self.next_loss_factor

            # # 2. detach prior ( train only the posterior )
            # prediction_z_copy = prediction_z
            # prediction_z_copy = prediction_z_copy.detach()

            # categroical_input = F.log_softmax(prediction_z_copy, dim=-1)
            # categorical_target = F.log_softmax(z, dim=-1)
            # phi_losses_posterior = F.kl_div(categroical_input, categorical_target, reduction='none', log_target=True)

            # phi_losses_posterior = phi_losses_posterior.sum(dim=-1) * target_padding_mask
            # return_dict['tokenwise_phi_losses_posterior'] = phi_losses_posterior
            # loss = phi_losses_posterior.sum() / target_padding_mask.sum()
            # return_dict['phi_loss_posterior'] = loss * self.next_loss_factor


            # log temperature
            return_dict["tokenwise_temperature"] = torch.ones(1)*self.quantizer.temperature

            # if self.decoder_mlp is not None:
            #     h_new = self.decoder_mlp(z_q)
            # else:
                # h_new = z_q

            l2_norm = F.mse_loss(h, h_new, reduction='none').sum(dim=-1) * target_padding_mask
            return_dict["tokenwise_reconstruction_loss"] = l2_norm
            recon_loss = l2_norm.sum() / padding_mask.sum()
            return_dict["reconstruction_loss"] = recon_loss * self.reconstruction_loss_factor
            if self.straight_through_eval and not self.training:
                h_new = h

            for i in range(self.quantizer.num_quantizers):
                ind = inds[i]
                encodings = F.one_hot(ind, self.quantizer.num_embeddings).float().reshape(-1, self.quantizer.num_embeddings)
                avg_probs = encodings.mean(0)
                
                #compute the codebook perplexity
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp() # exp(Entropy) = perplexity of a probabability distribution
                cluster_use = torch.sum(avg_probs > 0)
                return_dict[f"tokenwise_perplexity{i}"] = perplexity
                return_dict[f"tokenwise_cluster_use{i}"] = cluster_use

            return_dict["h"] = h_new
            return return_dict

    def setup_cache(
        self,
        batch_size: int,
        dtype: th.dtype,
        *,
        max_seq_len: int,
    ) -> None:
        if self.prior_prediction_attention is not None:
            self.prior_prediction_attention.setup_cache(batch_size, dtype, max_seq_len=max_seq_len)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are set up."""
        enabled = True
        if self.prior_prediction_attention is not None:
            enabled &= self.prior_prediction_attention.kv_cache is not None
        return enabled

    def reset_cache(self):
        """Reset the key value caches."""
        if self.prior_prediction_attention is not None:
            self.prior_prediction_attention.reset_cache()


class PHiLossCollector:
    def __init__(self):
        """
        A simple utility class for accumulating named losses.

        This class provides a straightforward way to collect and sum multiple loss
        values (e.g., PHi loss, self-critic loss) during a training or evaluation
        loop before they are logged or used for backpropagation.
        """
        self.losses = {}

    def add_loss(self, name: str, loss: torch.Tensor):
        """
        Adds a loss value to the running total for a given name.

        If the loss name does not already exist in the collector, it is
        initialized to zero before the new value is added.

        Args:
            name (str): The identifier for the loss.
            loss (torch.Tensor): The loss tensor to add.
        """
        if name not in self.losses:
            self.losses[name] = 0
        self.losses[name] += loss

    def reset(self):
        """Clears all accumulated losses."""
        self.losses = {}

class vae_encoder(nn.Module):
    """
    Deepmind encoder

    Args:
        input_channels: 
        n_hid: 
    """
    def __init__(self,
                codebook_dim : int = 512,
                tok_emb_dim: int = 768):
        super().__init__()

        self.net = nn.Linear(tok_emb_dim, codebook_dim, bias=False)

    def forward(self, x):
        return self.net(x)

class vae_decoder(nn.Module):
    """
    Deepmind encoder

    Args:
        n_init: 
        n_hid:
        output_channels: 
    """
    def __init__(self,
                 input_channels: int,
                 tok_emb_dim : int = 768):
        super().__init__()

        self.net = nn.Linear(input_channels, tok_emb_dim, bias=False)

    def forward(self, x):
        return self.net(x)

class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, 
                num_embeddings : int = 1024, 
                embedding_dim : int = 512):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self._kld_scale = 10.

        self.embed = nn.Embedding(num_embeddings, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z):
        """
        bsz: batch size
        s : sequence length
        hid_dim : hidden dimension
        """

        bsz, s, hid_dim = z.size()
        flatten = z.reshape(-1, self.embedding_dim)

        if not self.data_initialized.item() and self.training:
            print('running kmeans!')
            rp = torch.randperm(flatten.size(0) // 3 )
            #TODO: check for Warning: One of the clusters is empty. Re-run kmeans with a different initialization. 
            # return fun(*args, **kwargs)
            kd = kmeans2(flatten[rp].to(torch.float32).data.cpu().numpy(), 
                        self.num_embeddings, 
                        minit='points',)
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )

        _, ind = (-dist).max(1)
        ind = ind.view(bsz, s)

        z_q = self.embed_code(ind)
        commitment_cost = 0.25
        diff = commitment_cost * ( ( z_q.detach() - z ).pow(2) + ( z_q - z.detach() ).pow(2) ).mean()
        # diff *= self._kld_scale\

        z_q = z + (z_q - z).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    def __init__(self, num_embeddings: int = 1024, embedding_dim:int=512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.proj = nn.Linear(embedding_dim, num_embeddings, bias=False) # nn.Sequential(nn.Linear(embedding_dim, num_embeddings), nn.ReLU(inplace=True))

    def forward(self, logits):
        # TODO: temperature replace with 1
        one_hot = F.gumbel_softmax(logits, tau=1, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.embed.weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)

        qy = F.softmax(logits, dim=2)
        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.num_embeddings + 1e-10)

        return z_q, diff, ind


# class ResidualVectorQuantizer(nn.Module):
#     def __init__(self,
#                  posterior_mlp : nn.Module,
#                  decoder_mlp : nn.Module,
#                  num_embeddings_per_codebook: int = 1024,
#                  embedding_dim: int = 768,
#                  num_codebooks: int = 2,
#     ):
#         super().__init__()

#         self.num_codebooks = num_codebooks
#         self.num_embeddings = num_embeddings_per_codebook
#         self.embedding_dim = embedding_dim

#         self.residual_stacks = nn.ModuleList([
#             nn.Sequential(
#                 posterior_mlp, 
#                 GumbelQuantize(num_embeddings=num_embeddings_per_codebook, embedding_dim=embedding_dim),
#                 decoder_mlp,
#             ) for _ in range(self.num_codebooks)
#         ])

#     def forward(self, h):
#         latent_losses = []
#         zs = []
#         inds = []
#         residual_stream = h
#         h_sum = torch.zeros_like(h)
#         for m in self.residual_stacks:
#             z = m[:1](residual_stream)
#             z_q, latent_loss, ind = m[1:2](z)
#             h_prime = m[2:](z_q)
#             h_sum = h_sum + h_prime
#             residual_stream = residual_stream - h_prime
#             print('residual: ',torch.norm(residual_stream).item())
#             latent_losses.append(latent_loss)
#             inds.append(ind)
#             zs.append(z)
#         return h_sum, zs, latent_losses, inds

# class ResidualQuantizer(nn.Module):
#     def __init__(self,
#                  num_embeddings_per_codebook: int = 1024,
#                  embedding_dim: int = 768,
#                  num_codebooks: int = 2,):
#         super().__init__()
        
#         for _ in range(num_codebooks):
#             self.layers = nn.ModuleList([
#                 nn.Sequential(
#                     posterior(codebook_dim=num_embeddings_per_codebook, tok_emb_dim=embedding_dim),
#                     nn.Embedding(num_embeddings_per_codebook, embedding_dim),
#                     decoder(embedding_dim, embedding_dim),
#                 )
#             ] for _ in range(num_codebooks))
#         print(self.layers)

#     def forward(self, x):
#         residual = x
#         inds = []
#         for m in self.layers:
#             z = m[0](residual)
#             one_hot = F.gumbel_softmax(z, tau=1, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
#             ind = one_hot.argmax(dim=2)
#             zq = einsum('b s n, n d -> b s d', one_hot, m[1].weight) # shape: (bsz, num_tokens, embedding_dim)
#             xhat = m[2](zq)
#             residual = x - xhat
#             inds.append(ind)

#     def get_code(self, ind):
#         for i, m in enumerate(self.layers):
#             codes = m[1](ind[i])
#             print(codes.size())



# class posterior(nn.Module):
#     def __init__(self, codebook_dim : int = 1024, tok_emb_dim: int = 768):
#         super().__init__()
#         self.net = nn.Linear(tok_emb_dim, codebook_dim, bias=False)

#     def forward(self, x):
#         return self.net(x)

# class decoder(nn.Module):
#     def __init__(self, input_channels: int, tok_emb_dim : int):
#         super().__init__()
#         self.net = nn.Linear(input_channels, tok_emb_dim, bias=False)

#     def forward(self, x):
#         return self.net(x)

# helper function
def default(a, b):
    return a if a is not None else b 

class RQ(nn.Module):
    def __init__(self, num_quantizers, num_embedding, dim, 
                 posterior_mlp : nn.Module | None = None, 
                 decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # this is per codebook
        self.dim = dim

        self.codebooks = self._get_codebooks()
        self.posteriors = default(posterior_mlp, self._get_posteriors())
        # self.posterior = nn.Linear(self.dim, self.num_embeddings)
        # self.decoders = self._get_decoders()
        self.decoder = default(decoder_mlp, self._get_decoder())
        # for i in range(self.num_quantizers):

    def _get_codes_and_indices(self, x, codebook_idx):
        one_hot = F.gumbel_softmax(x, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebooks[codebook_idx].weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(x, dim=2)

        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.num_embeddings + 1e-10)
        return z_q, diff, ind
        
    # def _get_decoders(self):
    #     decoder = nn.ModuleList([nn.Linear(self.dim, self.dim)])
    #     for _ in range(1, self.num_quantizers):
    #         decoder.append(nn.Linear(self.dim, self.dim))
    #     return decoder
    
    def _get_decoder(self):
        shared_decoder = nn.Sequential(
            decoder(self.dim, 2048),
            decoder(2048, self.dim),)
        return shared_decoder
    #     for _ in range(1, self.num_quantizers):
    #         decoder.append(nn.Linear(self.dim, self.dim))
    #     return decoder
        

    def _get_posteriors(self):
        posterior = nn.ModuleList([
            nn.Sequential(encoder(self.dim, 2048), 
                          encoder(2048, self.num_embeddings),)
            ])
        for _ in range(1, self.num_quantizers):
            posterior.append(nn.Linear(self.dim, self.num_embeddings))
        return posterior

    def _get_codebooks(self):
        codebooks = nn.ModuleList([nn.Embedding(self.num_embeddings, self.dim)])
        for _ in range(1, self.num_quantizers):
            codebooks.append(nn.Embedding(self.num_embeddings, self.dim))
        return codebooks

    def forward(self, x):
        residual = x
        quantized_out = 0.
        self.clear_cache_zqs()
        
        indices = []
        latent_losses = []
        zs = []

        # TODO: modify posterior to handle both single module and module list 
        for i in range(self.num_quantizers):
            z = self.posteriors[i](residual)  # shape: (bsz, seq_len, num_embeddings)
            z_q, latent_loss, ind = self._get_codes_and_indices(z, i)

            self.cache_zqs(z_q)
            
            residual = residual - z_q.detach()
            
            quantized_out = quantized_out + z_q
            
            indices.append(ind)
            latent_losses.append(latent_loss)
            zs.append(z)
        
        quantized_out = self.decoder(quantized_out)
        indices = torch.stack(indices, dim=0)
        latent_losses = torch.stack(latent_losses, dim=0)
        zs = torch.stack(zs, dim=0)
        return quantized_out, latent_losses, indices, zs

    @property
    def get_quantized_inputs(self):
        return torch.stack(self._zqs)

    def cache_zqs(self, z_q):
        self._zqs.append(z_q)

    def clear_cache_zqs(self):
        self._zqs = [[] for _ in range(self.num_quantizers)][0]


class encoder(nn.Module):
    def __init__(self, dim: int = 768, num_embeddings : int = 1024):
        super().__init__()
        self.linear= nn.Linear(dim, num_embeddings)
        self.batch_norm = nn.BatchNorm1d(num_embeddings)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_dim : int = 768, out_dim : int = 768):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        return x