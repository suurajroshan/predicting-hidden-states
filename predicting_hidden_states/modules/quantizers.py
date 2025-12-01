import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from scipy.cluster.vq import kmeans2
from typing import List


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


        self._kld_scale = 1.

        self.embed = nn.Embedding(num_embeddings, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

        # TODO: temporary fix, resolve later
        self.num_quantizers = 1

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
            print(flatten.size())
            rp = torch.randperm(flatten.size(0))
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
        diff = commitment_cost * ( ( z_q.detach() - z ).pow(2) + ( z_q - z.detach() ).pow(2) )
        diff *= self._kld_scale

        z_q = z + (z_q - z).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        return z_q, diff, [ind]

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
        one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.embed.weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)

        qy = F.softmax(logits, dim=2)
        # KL term from the ELBO with a uniform prior
        diff = qy * torch.log(qy * self.num_embeddings + 1e-10)

        return z_q, diff, ind

# # helper function
# def default(a, b):
#     return a if a is not None else b

class ResidualQuantizeStep(nn.Module):
    def __init__(self, dim , num_embeddings, stage, shared_encoder=None, shared_decoder=None):
        super().__init__()

        self.dim = dim
        self.num_embeddings = num_embeddings

        # if not shared_encoder:
        self.posterior = nn.Sequential(encoder(self.dim, 2048), 
                        encoder(2048, self.num_embeddings))
        
        self.codebook = nn.Embedding(num_embeddings, dim)
        # self.codebook.weight.data.normal_(mean=0.0, std=1/(2**stage))
        
        # if not shared_decoder:
        #     self.decoder = nn.Sequential(decoder(self.dim, 1024),
        #                                  decoder(1024, self.dim))

    def encode(self, x, shared_encoder: torch.nn.Module | None = None):
        posterior = default(shared_encoder, self.posterior)
        z = posterior(x)
        return z

    # def decode(self, z_q, shared_decoder: torch.nn.Module | None = None):
    #     decoder = default(shared_decoder, self.decoder)
    #     x_tilde = decoder(z_q)
    #     return x_tilde

    def quantize(self, z):
        assert self.temperature is not None
        one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebook.weight) # shape: (bsz, num_tokens, embedding_dim)        
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(z, dim=2)

        # KL term from the ELBO with a uniform prior
        latent_loss = qy * torch.log(qy * self.num_embeddings + 1e-10)
        return z_q, latent_loss, ind

    def forward(self, res, z_hat, shared_encoder=None, shared_decoder=None):
        z = self.encode(res, shared_encoder=shared_encoder)
        z_q, latent_loss, code_idxs = self.quantize(z)
        res = res  - z_q.detach()
        z_hat = z_hat + z_q
        # x_tilde = self.decode(z_hat, shared_decoder=shared_decoder)
        return res, code_idxs, z, z_hat # , x_tilde


class ResidualQuantize(nn.Module):
    def __init__(self, 
                 num_quantizers : int, 
                 num_embedding : int, 
                 dim : int, 
                 shared_encoder_flag : bool = False,
                 shared_decoder_flag : bool = True):
                #  posterior_mlp : nn.Module | None = None, 
                #  decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # embeddings per codebook
        self.dim = dim

        self.shared_encoder = None
        self.shared_decoder = None
        # if shared_encoder_flag:
        #     self.shared_encoder = nn.Sequential(encoder(self.dim, 2048), 
        #                                         encoder(2048, self.num_embeddings))
        # if shared_decoder_flag:
        #     self.shared_decoder = nn.Sequential(decoder(self.dim, 1024),
        #                                         decoder(1024, self.dim))
        self.shared_decoder = nn.Sequential(decoder(self.dim, 1024),
                                                decoder(1024, self.dim))
        
        self.stages = []
        for m in range(self.num_quantizers):
            rq_step = ResidualQuantizeStep(self.dim, self.num_embeddings, m)
            self.add_module(f"stage{m}", rq_step)
            self.stages.append(rq_step)

    def forward(self, x):
        residual = x
        z_hat = 0.
        # latent_losses = torch.zeros(self.num_quantizers, x.shape[0])

        reconstruction_losses_per_stage = torch.zeros(self.num_quantizers, *x.shape).to(x.device)
        self.zs = torch.zeros(self.num_quantizers, *x.shape[:-1], self.num_embeddings).to(x.device)
        self.indices = torch.zeros_like(self.zs[..., 0], dtype=torch.long).to(x.device)
        losses = {}
        for m, stage in enumerate(self.stages):
            stage.temperature = self.temperature
            
            residual, code_idxs, z, z_q_summed = stage(residual, z_hat, self.shared_encoder, self.shared_decoder)
            
            # reconstruction_losses_per_stage[m] = (x_hat - x).pow(2)
            self.zs[m, ...] = z
            self.indices[m, ...] = code_idxs
        x_hat = self.shared_decoder(z_q_summed)
        # losses['reconstruction_losses'] = reconstruction_losses_per_stage
        return x_hat, residual, losses

    @property
    def get_zs(self):
        return self.zs

    @property
    def get_code_indices(self):
        return self.indices

    # @property
    # def get_quantized_inputs(self):
    #     return torch.stack(self._zqs)

    # def cache_zqs(self, z_q):
    #     self._zqs.append(z_q)

    # def clear_cache_zqs(self):
    #     self._zqs = [[] for _ in range(self.num_quantizers)][0]

# helper function
def default(a, b):
    return a if a else b

class TestRQ(nn.Module):
    def __init__(self, num_quantizers, num_embedding, dim, 
                 posterior_mlp : nn.Module | None = None, 
                 decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # this is per codebook
        self.dim = dim

        self.codebooks = self._init_codebooks()
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
            decoder(self.dim, self.dim*2),
            decoder(self.dim*2, self.dim),)
        return shared_decoder
    #     for _ in range(1, self.num_quantizers):
    #         decoder.append(nn.Linear(self.dim, self.dim))
    #     return decoder

    def _get_posteriors(self):
        posterior = nn.ModuleList([
            nn.Sequential(encoder(self.dim, self.dim*2), 
                          encoder(self.dim*2, self.num_embeddings),)
            ])
        for _ in range(1, self.num_quantizers):
            posterior.append(nn.Linear(self.dim, self.num_embeddings))
        return posterior

    def _init_codebooks(self):
        codebooks = nn.ModuleList([nn.Embedding(self.num_embeddings, self.dim)])
        for m in range(1, self.num_quantizers):
            cb_init = nn.Embedding(self.num_embeddings, self.dim)
            cb_init.weight.data.normal_(mean=0.0, std=1/(2**m))
            codebooks.append(cb_init)
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

    def save_codebooks(self, path = f'checkpoints/codebooks', filename='last.pt'):
        filepath = os.path.join(path, filename)
        assert os.path.exists(filepath)
        if not os.path.exists(path):
            print('creating directory for saving codebooks')
        os.makedirs(path, exist_ok=True)
        torch.save(self.codebooks, filepath)
        print(f'saved codebooks at {filepath}')

class RQStep(nn.Module):
    def __init__(self, m, posteriors, codebooks, decoders):
        super().__init__()
        if type(posteriors) == nn.ModuleList:
            self.posterior = posteriors[m]
        else:
            self.posterior = posteriors
        if type(decoders) == nn.ModuleList:
            self.decoder = decoders[m]
        else:
            self.decoder = decoders

        if type(codebooks) == nn.ModuleList:
            self.codebook = codebooks[m]
        else:   
            self.codebook = codebooks

    def forward(self, res):
        z = self.posterior(res)
        print(self.temperature)
        one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=True) # shape: (bsz, num_tokens, num_embeddings)
        z_q = einsum('b s n, n d -> b s d', one_hot, self.codebook.weight) # shape: (bsz, num_tokens, embedding_dim)
        ind = one_hot.argmax(dim=2)
        qy = F.softmax(z, dim=2)

        # KL term from the ELBO with a uniform prior
        print(f'num embeddings: {self.codebook.weight.shape[0]}')
        diff = qy * torch.log(qy * self.codebook.weight.shape[0] + 1e-10)
        return z, z_q, diff, ind

class TestResQt(nn.Module):
    def __init__(self, num_quantizers, num_embedding, dim, 
                 posterior_mlp : nn.Module | None = None, 
                 decoder_mlp : nn.Module | None = None):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embedding # this is per codebook
        self.dim = dim

        self.codebooks = self._init_codebooks()
        self.posteriors = default(posterior_mlp, self._get_posteriors())
        # self.posterior = nn.Linear(self.dim, self.num_embeddings)
        # self.decoders = self._get_decoders()
        self.decoder = default(decoder_mlp, self._get_decoder())
        # for i in range(self.num_quantizers):
        self.stages = []
        for m in range(self.num_quantizers):
            rq_step = RQStep(m, self.posteriors, self.codebooks, self.decoder)
            self.add_module(f"stage{m}", rq_step)
            self.stages.append(rq_step)



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
            decoder(self.dim, self.dim*2),
            decoder(self.dim*2, self.dim),)
        return shared_decoder
    #     for _ in range(1, self.num_quantizers):
    #         decoder.append(nn.Linear(self.dim, self.dim))
    #     return decoder

    def _get_posteriors(self):
        posterior = nn.ModuleList([
            nn.Sequential(encoder(self.dim, self.dim*2), 
                          encoder(self.dim*2, self.num_embeddings),)
            ])
        for _ in range(1, self.num_quantizers):
            posterior.append(nn.Linear(self.dim, self.num_embeddings))
        return posterior

    def _init_codebooks(self):
        codebooks = nn.ModuleList([nn.Embedding(self.num_embeddings, self.dim)])
        for m in range(1, self.num_quantizers):
            cb_init = nn.Embedding(self.num_embeddings, self.dim)
            codebooks.append(cb_init)
        return codebooks

    def forward(self, x):
        residual = x
        quantized_out = 0.
        self.clear_cache_zqs()
        
        indices = []
        latent_losses = []
        zs = []

        # TODO: modify posterior to handle both single module and module list 
        for stage in self.stages:
            stage.temperature = self.temperature
            z, z_q, latent_loss, ind = stage(residual)    
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

    def save_codebooks(self, path = f'checkpoints/codebooks', filename='last.pt'):
        filepath = os.path.join(path, filename)
        assert os.path.exists(filepath)
        if not os.path.exists(path):
            print('creating directory for saving codebooks')
        os.makedirs(path, exist_ok=True)
        torch.save(self.codebooks, filepath)
        print(f'saved codebooks at {filepath}')



# class encoder(nn.Module):
#     def __init__(self, dim: int = 768, num_embeddings : int = 1024):
#         super().__init__()
#         self.linear= nn.Linear(dim, num_embeddings)
#         self.batch_norm = nn.BatchNorm1d(num_embeddings)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.linear(x)
#         # x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
#         # x = self.relu(x)
#         return x

# class decoder(nn.Module):
#     def __init__(self, in_dim : int = 768, out_dim : int = 768):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.batch_norm = nn.BatchNorm1d(out_dim)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.linear(x)
#         # x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
#         # x = self.relu(x)
#         return x

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

##### QINCO Quantizer #####
class QincoStep(nn.Module):
    def __init__(self, 
                 num_embeddings: int = 1024, 
                 embedding_dim:int = 512, 
                 num_residual_blocks:int=2,
                 res_block_hidden_dim: int = 1024,
                 init_codebook = None):
        super().__init__()
        assert init_codebook is not None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_residual_blocks = num_residual_blocks
        self.h_dim = res_block_hidden_dim

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data = init_codebook.weight.data.clone()
        self.concat_mlp = nn.Linear(2*embedding_dim, embedding_dim)

        self.residual_blocks = []
        for i in range(num_residual_blocks):
            residual_block = nn.Sequential(nn.Linear(embedding_dim, res_block_hidden_dim, bias=False),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(res_block_hidden_dim, embedding_dim, bias=False),)
            self.add_module(f"residual_block_{i}", residual_block)
            self.residual_blocks.append(residual_block)

    def encode(self, x_tilde, x):
        zqs = self.codebook.weight
        bsz, seq_len, dim = x.shape

        zqs_r = zqs.repeat(bsz, 1, 1).reshape(-1, self.embedding_dim)
        x_tilde_r = x_tilde[:, None, ...].repeat(1, self.num_embeddings, 1).reshape(-1, self.embedding_dim)

        cc = torch.cat((zqs_r, x_tilde_r), dim=1)
        zqs_r = zqs_r + self.concat_mlp(cc)

        for res_block in self.resiual_blocks:
            zqs_r = zqs_r + res_block(zqs_r)

        zqs_r = zqs_r.reshape(bsz, self.num_embeddings, self.embedding_dim) + x_tilde.reshape(bsz, 1, self.embedding_dim)
        codes, x_tilde_next = assign_batch_multiple(x, zqs_r)

        return codes, x_tilde_next - x_tilde
    
    def decode(self, x_tilde, codes):
        zqs = self.codebook(codes)
        cc = torch.cat((zqs, x_tilde), dim=1)
        zqs = zqs + self.concat_mlp(cc)

        for res_block in self.residual_blocks:
            zqs = zqs + res_block(zqs)

        return zqs
    
class QincoQuantize(nn.Module):
    def __init__(self, num_embeddings: int = 1024, 
                 embedding_dim : int = 512, 
                 num_quantizers : int = 3,
                 init_codebooks = None):
        super().__init__()

        assert init_codebooks is not None

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers

        self.codebook0 = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook0.weight.data = init_codebooks[0].weight.data.clone()

        self.stages = []
        for m in range(1, num_quantizers):
            stage = QincoStep(init_codebook=init_codebooks[m])
            self.add_module(f"stage{m}", stage)
            self.stages.append(stage)
    
    def decode(self, codes):
        x_tilde = self.codebook0(codes[0])
        for i, stage in enumerate(self.stages):
            x_tilde = x_tilde + stage.decode(x_tilde, codes[i+1])
        return x_tilde
    
    def encode(self, x):
        bsz, seq_len, dim = x.shape
        codes = torch.zeros(bsz, self.num_quantizers, dtype=int, device=x.device)
        
        code0 = assign_codes(x, self.codebook0.weight)
        codes[:,0] = code0

        x_tilde = self.codebook0(code0)

        for i, stage in enumerate(self.stages):
            codes[:, i+1], toadd = stage.encode(x_tilde, x)
            x_tilde = x_tilde + toadd
        return codes, x_tilde 

    def forward(self, x):
        codes, x_tilde = self.encode(x)
        losses = torch.zeros(self.num_quantizers)
        x_tilde = self.codebook0(codes[:,0])
        losses[0] = (x_tilde - x).pow(2).sum()
        for i, stage in enumerate(self.stages):
            x_tilde = x_tilde + self.decode(x_tilde, codes[:, i+1])
            losses[i+1] = (x_tilde - x).pow(2).sum()
        return codes, x_tilde, losses
    

def quantizer_module(self_prediction_information_bottleneck,
                         self_prediction_module):
    quantizer_module = {
            'vector_quantize': VQVAEQuantize,
            'gumbel_quantize': GumbelQuantize,
            'residual_quantize': TestRQ, # TestResQt, # ResidualQuantize,
            'residual_quantize_qinco': QincoQuantize,
            'continuous': None,
        }[self_prediction_information_bottleneck]

    if self_prediction_information_bottleneck == 'residual_quantize':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        num_quantizers = self_prediction_module["num_quantizers"]
        # shared_encoder_flag = self_prediction_module["shared_encoder"]
        # shared_decoder_flag = self_prediction_module["shared_decoder"]
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        # quantizer_mlp = quantizer_module(num_embeddings=codebook_dim, embedding_dim=codeword_dim)
        posterior_mlp = encoder(codebook_dim, codeword_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        quantizer_mlp = quantizer_module(num_quantizers, codebook_dim, codeword_dim,) #  shared_encoder_flag, shared_decoder_flag)
    
    elif self_prediction_information_bottleneck == 'vector_quantize':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        # quantizer_mlp = quantizer_module(num_embeddings=codebook_dim, embedding_dim=codeword_dim)
        posterior_mlp = encoder(codeword_dim, codeword_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        quantizer_mlp = quantizer_module(codebook_dim, codeword_dim)

    elif self_prediction_information_bottleneck == 'residual_quantize_qinco':
        codeword_dim = self_prediction_module['codeword_dim']
        codebook_dim = self_prediction_module['codebook_dim']
        num_quantizers = self_prediction_module["num_quantizers"]
        assert self_prediction_module["qinco"]["saved_codebooks"] is not None or ''
        print('loading pretrained codebooks from ', self_prediction_module["qinco"]["saved_codebooks"])
        saved_codebooks = torch.load(self_prediction_module["qinco"]["saved_codebooks"], weights_only=False)
        embed_dim = codeword_dim  # override embed dim if using quantization
        hidden_dim = codeword_dim * 8 // 3 # TODO: currrently a heuristic, can be parameterized later
        print(f'number of embeddings: {codebook_dim}')
        print(f'embedding dimension: {codeword_dim}')
        # quantizer_mlp = quantizer_module(num_embeddings=codebook_dim, embedding_dim=codeword_dim)
        posterior_mlp = encoder(codebook_dim, codeword_dim)
        decoder_mlp = decoder(codeword_dim, embed_dim)
        quantizer_mlp = quantizer_module(num_quantizers, codebook_dim, codeword_dim, saved_codebooks) #  shared_encoder_flag, shared_decoder_flag)
    
    else:
        posterior_mlp = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        quantizer_mlp = quantizer_module
        decoder_mlp=nn.Linear(embed_dim, embed_dim, bias=False)

    return posterior_mlp, quantizer_mlp, decoder_mlp