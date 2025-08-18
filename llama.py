from dataclasses import dataclass, field
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
import time

# When converting GPT to Llama we have to replace layer norm with Root Mean Squared Norm RMSNorm
# We have to change MultiHeadedAttention with Grouped Query Attention
# We have to implement Rope embeddings - inside GQA
# Change to SiLU activation from GeLU, add additional layer SwiGLU
# Weight sharing between token embedding layer and last linear layer

@dataclass
class RopeConfig:
    """
    This class is used for RoPE scalling params
    """
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_context_length: int = 8192


@dataclass
class LlamaConfig:
    """
    This class is going to be used to store config parameters for Llama 3.2 model. Defaults to 1B model.
    """
    vocab_size: int = 128256 # Vocabulary size
    context_length: int = 131072 # Supported context lenght
    emb_dim: int = 2048 # Token embedding dimension
    n_heads: int = 32 # Number of self attention heads in a single transformer block (layer)
    n_layers: int = 16 # Number of transformer blocks (called layers in Llama)
    hidden_dim: int = 8192 # Intermediate dimensions used in MLP
    n_kv_groups: int = 8 # Number of kv groups
    rope_base: float = 500000.0  # Used to determine freqency for RoPE
    rope_freq: RopeConfig = field(default_factory=RopeConfig) # RoPE related config
    dtype: torch.dtype = torch.bfloat16
    


class RMSNormLayer(nn.Module):
    """
    This class represents implementation of RMSNorm layer.
    PyTorch offers layer RMSNorm layer but I will implement it from scratch for educational purposes 
    More can be found here: https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    """
    def __init__(self, embd_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.embd_dim = embd_dim
        self.weight = nn.Parameter(torch.ones(embd_dim)).float()

    def forward(self, x: torch.Tensor):
        root_mean_squares = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        x_normalized = x / root_mean_squares
        return (x_normalized * self.weight).to(dtype=x.dtype)



class MLP(nn.Module):
    """
    This class represents MLP that is going to be used inside the transformer block
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False, dtype=config.dtype)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, bias=False, dtype=config.dtype)
    
    def forward(self, x: torch.Tensor):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# I need to precompute params necessary for RoPE
def _precompute_rope_params(head_dim, config:LlamaConfig):
    # intiger -> binary -> sinusoidal -> RoPE
    # RoPE does not polute the semantics of token embeddings because we do not add vector containing the positional info, we rotate queries and keys
    # Encodes both absolute and relative position information
    # theta = position * omega
    # omega = 1 / 500000 ^ (2i/dim)
    # rot matrix = [[cos(theta) -sin(theta)] [sin(theta) cos(theta)]]
    # Token embedding gets split into groups of 2 so there are dim // 2 groups of 2 that get rotated
    # Lower indeces in token embeddings oscilate more quickly capturing small changes
    assert head_dim % 2 == 0, "Head dimension must be divisible by 2"
    position = torch.arange(0, config.context_length, dtype=torch.float32).view(config.context_length, 1) # converting tensor of rank 1 to -> (context_lenght, 1)
    i = torch.arange(0, head_dim, 2, dtype=torch.float32) # (head_dim/2,) -> Number of groups (pairs of 2) in attention head [x0, x1, x2, x3] -> (x0, x1), (x2, x3)
    omega = 1.0 / torch.pow(config.rope_base, (i / head_dim))
    # Scaling frequencies - YaRN paper - this helps with long sequences
    if config.rope_freq is not None:
        low_freq_wavelen = config.rope_freq.original_context_length / config.rope_freq.low_freq_factor
        high_freq_wavelen = config.rope_freq.original_context_length / config.rope_freq.high_freq_factor

        wavelen = 2 * torch.pi / omega #
        
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, omega / config.rope_freq.factor, omega
        )

        smooth_factor = (
            config.rope_freq.original_context_length / wavelen
            - config.rope_freq.low_freq_factor
        ) / (
            config.rope_freq.high_freq_factor - config.rope_freq.low_freq_factor
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (omega / config.rope_freq.factor)
            + smooth_factor * omega
        )

        is_medium_freq  = (wavelen <= low_freq_wavelen) & (wavelen > high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        omega = inv_freq_llama
    
    omega = omega.view(1, int(head_dim/2)) # (1, head_dim/2)
    theta = position * omega # (context_length, head_dim/2) -> matrix that contains angle for each position and group
    theta = torch.cat((theta, theta), dim=1) # (context_length, head_dim) -> matrix that contains angle for each position and index in attention head
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    
    return cos, sin

# I need to define apply_rope fn, this will be used inside my GQA to rotate queries and keys
def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be divisible by 2"
    # Split x into two even parts on head_dim
    x_first_half = x[..., : head_dim // 2] # real part
    x_second_half = x[..., head_dim // 2 :] # imaginary part
    cos = cos[start_pos:start_pos+seq_len, :].reshape((1, 1, seq_len, -1))
    sin = sin[start_pos:start_pos+seq_len, :].reshape((1, 1, seq_len, -1))
    # o1' = cos * x_first_half - x_second_half * sin 
    # o2' = sin * x_first_half + x_second_half * cos
    # This is same as if we multiplied imaginary number -> Rotation in 2D plain can be expressed as multiplication of complex number by e^iomega (complex number with magnitude of 1)
    # z = a + ib with e^(i * omega) -> (a + ib)(cos(omega) + i*sin(omega)) = a * cos(omega) + a*i*sin(omega) + b*i*cos(omega) - b * sin(omega)
    # (a * cos(omega) - b * sin(omega)) + i(a*sin(omega) + b * cos(omega)) = cos(omega_prim + omega) + isin(omeg_prim + omega)
    rotated = torch.cat((-x_second_half, x_first_half), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
       
    return x_rotated.to(dtype=x.dtype)
    

class GroupedQueryAttention(nn.Module):
    """
    This class represents implementation of Gropued Query Attention Mechanism found in Llama 3.2 model
    GQA reduces the number of parameters needed by sharing Keys and Values between attenation heads
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.d_out = config.emb_dim
        self.num_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_heads // config.n_kv_groups
        self.W_key = nn.Linear(config.emb_dim, self.num_kv_groups * self.head_dim, bias=False, dtype=config.dtype) # add dtypes
        self.W_value = nn.Linear(config.emb_dim, self.num_kv_groups * self.head_dim, bias=False, dtype=config.dtype)
        self.W_query = nn.Linear(config.emb_dim, self.d_out, bias=False, dtype=config.dtype)
        self.out_proj = nn.Linear(self.d_out, self.d_out, bias=False, dtype=config.dtype)
        # kv_cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self,x: torch.Tensor,
                sin: torch.Tensor,cos: torch.Tensor,
                mask: torch.Tensor, use_cache: bool = False, start_pos = 0):
        batch_size, seq_length, d_in = x.shape
        
        # Generating Q, K and V
        q = self.W_query(x) # (batch_size, seq_length, d_out)
        k = self.W_key(x) # (batch_size, seq_length, self.num_kv_groups * self.head_dim)
        v = self.W_value(x) # (batch_size, seq_length, self.num_kv_groups * self.head_dim)


        # Reshaping
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_groups, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_groups, self.head_dim)

        q = q.transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim)
        k = k.transpose(1, 2) # (batch_size, num_kv_groups, seq_length, head_dim)
        v = v.transpose(1, 2) # (batch, num_kv_groups, seq_length, head_dim)

        # RoPE Application -> Rotating queries and keys
        if cos is not None:
            q = _apply_rope(q, cos, sin, start_pos)
            k = _apply_rope(k, cos, sin, start_pos)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                # x shape (batch_size, 1, embd_dim)
                # q shape (batch_size, 1, d_out)
                # q after reshaping (batch_size, num_heads, seq_length, head_dim)
                # k and v shape (batch_size, seq_len, self.num_kv_groups * self.head_dim)
                # k and v after reshaping (batch_size, seq_len, num_kv_groups, head_dim) after reshaping
                self.cache_k = torch.cat((self.cache_k, k), dim=2)
                self.cache_v = torch.cat((self.cache_v, v), dim=2)
                
                # Now we take all the values from cache so we can get complete K and V matrices for 
                k = self.cache_k
                v = self.cache_v
        

        k = k.repeat_interleave(self.group_size, dim=1) # (batch_size, num_kv_groups * group_size = num_heads, seq_length, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1) # (batch_size, num_heads, seq_length, head_dim)
        
        attn_scores = (q @ k.transpose(2, 3) * (1 / (k.size(-1) ** 0.5))) # (QK^T)/sqrt(dim_k)
        
        # Applying mask for Causal Language Modeling
        attn_scores.masked_fill(mask, -torch.inf)
        
        # Normalizing attention scores by applying row-wise softmax -> attention in each row sums up to 1
        normalized_attn_scores = F.softmax(attn_scores, dim=-1)

        # (batch_size, num_heads, seq_length, seq_length) x (batch_size, num_heads, seq_length, head_dim) = (batch_size, num_heads, seq_length, head_dim)
        y = normalized_attn_scores @ v
 
        # Transposing (batch_size, num_heads, seq_length, head_dim) to (batch_size, seq_length, num_heads, head_dim)
        # Reshaping output tensor (batch_size, seq_length, num_heads, head_dim) so we get (batch_size, seq_length, d_in)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, d_in)
        
        y = self.out_proj(y)
        
        return y
    
    def reset_kv_cache(self):
        self.cache_k = None
        self.cache_v = None


class TransformerBlock(nn.Module):
    """
    This class represents transformer block inside Llama 3.2 architecture
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNormLayer(config.emb_dim, epsilon=1e-5)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNormLayer(config.emb_dim, epsilon=1e-5)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, use_cache: bool = False, start_pos = 0):
        x = x + self.self_attn(self.input_layernorm(x), sin, cos, mask, use_cache=use_cache, start_pos=start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
    
    def reset_kv_cache(self):
        self.self_attn.reset_kv_cache()


class Llama(nn.Module):
    """
    This class represents the complete Llama 3.2 model architecture
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]) # This could be changed to nn.ModuleDict based on Unsloth state_dict
        self.norm = RMSNormLayer(config.emb_dim, epsilon=1e-5)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)
        cos, sin = _precompute_rope_params(
            head_dim=config.emb_dim // config.n_heads,
            config=config
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # lm_head shares the weight matrix with embed_token
        # This is quite clever optimization as it turns out it helps with regularization and reduces the number of trainable parameters
        # Additionally this does not seem correct at the first glance since embed_tokens = nn.Embedding(vocab_size, embed_dim) and lm_head = nn.Linear(embed_dim, vocab_size)
        # But since lm_head is linear it will be saved as (vocab_size, embed_dim) actually and that matches with dimensions of embed_tokens
        # More info can be found here: 
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
        self.embed_tokens.weight = self.lm_head.weight
        self.current_position = 0

    def forward(self, input: torch.Tensor, use_cache: bool = False):
        # input has shape (number of batches, number of tokens in sequence)
        x = self.embed_tokens(input) # Shape (batch_size, sequence_length, embed_dim)
        
        seq_length = x.shape[1]
        
        start_pos = self.current_position
        if use_cache:
            self.current_position += seq_length 
            mask = torch.triu(
                torch.ones(
                    seq_length,
                    seq_length,
                    device=x.device,
                    dtype=torch.bool), diagonal=1
                    )[start_pos:self.current_position, :self.current_position]

        else:
            mask = torch.triu(
                torch.ones(
                    seq_length,
                    seq_length,
                    device=x.device,
                    dtype=torch.bool), diagonal=1
                    )
        
        for layer in self.layers:
            x = layer(x, mask, self.cos, self.sin, use_cache, start_pos)
        
        x = self.norm(x) # Shape (batch_size, sequence_length, embed_dim)
        
        logits = self.lm_head(x) # Shape (batch_size, sequence_length, vocab_size)
    
        return logits
    
    def reset_kv_cache(self):
        for layer in self.layers:
            layer.reset_kv_cache()
        self.current_pos = 0
    

    def _assign(self, left: torch.Tensor, right: torch.Tensor, tensor_name: str):
        if left.shape != right.shape:
            raise ValueError(
                f"""Shape mismatch in tensor: {tensor_name}
                Left: {left.shape} - Right: {right.shape}""")
        
        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))
        
    
    def _load_weights(self, config: LlamaConfig, params: torch.Tensor):
        self.embed_tokens.weight = self._assign(self.embed_tokens.weight,
                                                params["model.embed_tokens.weight"],
                                                "model.embed_tokens.weight")
        
        for layer in range(config.n_layers):
            # Input Layer Normalization Weights
            self.layers[layer].input_layernorm.weight = self._assign(
                self.layers[layer].input_layernorm.weight,
                params[f"model.layers.{layer}.input_layernorm.weight"],
                f"model.layers.{layer}.input_layernorm.weight"
            )
            # Self Attention weights
            self.layers[layer].self_attn.W_query.weight = self._assign(
                self.layers[layer].self_attn.W_query.weight,
                params[f"model.layers.{layer}.self_attn.q_proj.weight"],
                f"model.layers.{layer}.self_attn.q_proj.weight"
            )

            self.layers[layer].self_attn.W_key.weight = self._assign(
                self.layers[layer].self_attn.W_key.weight,
                params[f"model.layers.{layer}.self_attn.k_proj.weight"],
                f"model.layers.{layer}.self_attn.k_proj.weight"
            )

            self.layers[layer].self_attn.W_value.weight = self._assign(
                self.layers[layer].self_attn.W_value.weight,
                params[f"model.layers.{layer}.self_attn.v_proj.weight"],
                f"model.layers.{layer}.self_attn.v_proj.weight"
            )
            
            self.layers[layer].self_attn.out_proj.weight = self._assign(
                self.layers[layer].self_attn.out_proj.weight,
                params[f"model.layers.{layer}.self_attn.o_proj.weight"],
                f"model.layers.{layer}.self_attn.o_proj.weight"
            )
            # Post Attention Norm Layer
            self.layers[layer].post_attention_layernorm.weight = self._assign(
                self.layers[layer].post_attention_layernorm.weight,
                params[f"model.layers.{layer}.post_attention_layernorm.weight"],
                f"model.layers.{layer}.post_attention_layernorm.weight"               
            )  
            # MLP weights
            self.layers[layer].mlp.gate_proj.weight = self._assign(
                self.layers[layer].mlp.gate_proj.weight,
                params[f"model.layers.{layer}.mlp.gate_proj.weight"],
                f"model.layers.{layer}.mlp.gate_proj.weight"               
            )

            self.layers[layer].mlp.up_proj.weight = self._assign(
                self.layers[layer].mlp.up_proj.weight,
                params[f"model.layers.{layer}.mlp.up_proj.weight"],
                f"model.layers.{layer}.mlp.up_proj.weight"               
            )
            
            self.layers[layer].mlp.down_proj.weight = self._assign(
                self.layers[layer].mlp.down_proj.weight,
                params[f"model.layers.{layer}.mlp.down_proj.weight"],
                f"model.layers.{layer}.mlp.down_proj.weight"               
            )
        
        self.norm.weight = self._assign(self.norm.weight,
                                                params["model.norm.weight"],
                                                "model.norm.weight")
        
        # Weight sharing
        self.lm_head.weight = self._assign(self.lm_head.weight,
                                                params["model.embed_tokens.weight"],
                                                "model.embed_tokens.weight")

    def _download_weights(self):
        weights_file = hf_hub_download(
            repo_id=f"unsloth/Llama-3.2-1B-Instruct",
            filename="model.safetensors",
            local_dir=f"Llama-3.2-1B"
        )
        return load_file(weights_file)


    def _transfer_model_to_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        self.to(device)


    def setup_model(self, device_auto: bool = False):
        combined_weights = self._download_weights()
        self._load_weights(self.config, combined_weights)
        
        if device_auto:
            self._transfer_model_to_device()

    
    def _calc_total_params(self):
        """
        Method that calculates total number of parameters
        """
        # Should i substract params from the last layer (first layer because of the w sharing)
        return sum(p.numel() for p in self.parameters())
    
    
    def _model_memory_size(self, dtype=torch.float32):
        """
        Method that calculates model memory size
        """
        total_params = 0
        total_grads = 0

        for param in self.parameters():
            param_size = param.numel()
            total_params += param_size

            if param.requires_grad:
                total_grads += param_size
        # Non parameters that require memory
        # Substract params form the last layer because they are shared with first layer - TODO
        total_buffers = sum(buf.numel() for buf in self.buffers())
        element_size = torch.tensor(0, dtype=dtype).element_size()
        total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

        total_memory_gb = total_memory_bytes / (1024**3)

        return total_memory_gb
    

    def describe(self, dtype=torch.float32):
        total_params = self._calc_total_params()
        total_memory_gb = self._model_memory_size(dtype=dtype)
        print(f"""Total number of parameters: {total_params}
              Total memory requirements: {total_memory_gb:.2f} GB
              Type: {dtype}""")
        


# Model Initialization
llama_32_1B_config = LlamaConfig()
model = Llama(llama_32_1B_config)

model.setup_model(device_auto=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.describe()

# Generation
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Adding batch dim since our model receives B,S,ED
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove B dim
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, eos_id=128009, use_cache=False):

    if use_cache:
        model.reset_kv_cache()

        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond, use_cache=use_cache) 
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            if next_idx == eos_id:
                break
            idx = torch.cat([idx, next_idx], dim=1)
            with torch.no_grad(): 
                logits = model(next_idx, use_cache=use_cache)
        
        return idx

    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating if eos token encountered
            break

        idx = torch.cat((idx, idx_next), dim=1)  # Appending newly predicted token to a sequence for each batch (batch_size, num_tokens+1)

    return idx

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

messages = [{"role": "user", "content": "Who is Van Gogh?"}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

idx = text_to_token_ids(prompt, tokenizer).to(device)


start = time.time()
generations = generate(model, idx, 100, 8192, use_cache=True)
end = time.time()

print(end-start)

print(token_ids_to_text(generations, tokenizer))
