import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', type=str, help='path to the torch model.')
parser.add_argument('--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('--batch_size', type=int, default=1, help="sequence length")

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, fp32=True, device_map="auto").eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.transformer
layers = transformer.h
BATCH_SIZE          =   args.batch_size
SEQ_LENGTH          =   args.seq_length
NUM_LAYERS          =   config.num_hidden_layers
HIDDEN_SIZE         =   config.hidden_size
NUM_ATTENTION_HEADS =   config.num_attention_heads
HEAD_DIM            =   HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE          =   config.vocab_size
MAX_POS_EMB         =   config.max_position_embeddings


def _rotate_half(x):
    from einops import rearrange
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    """ Apply rotary embedding to the first rotary_dim of the iput

    Arguments:
      t (tensor(batch_size, seq_len, n_head, head_dim)):
        the input embedding/hidden states
      freqs (list[tensor(1, seq_len, 1, rotary_dim), tensor(1, seq_len, 1, rotary_dim)]):
        the cached cos/sin position embeddings
    """
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_float = t.float()
    t_rot, t_pass = t_float[..., :rot_dim], t_float[..., rot_dim:]
    t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)
    return torch.cat((t_rot, t_pass), dim=-1).type_as(t)

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.wte(input_ids)
        return out.float()

class QwenPreAttnBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(MAX_POS_EMB)
        self.cos_emb = self.rotary_emb[0].view(MAX_POS_EMB, HEAD_DIM)
        self.sin_emb = self.rotary_emb[1].view(MAX_POS_EMB, HEAD_DIM)

    def forward(self, hidden_states, position_ids):
        bsz, sqlen, _ = hidden_states.size()
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)

        layernorm_output = self.layer.ln_1(hidden_states)
        mixed_x_layer = self.layer.attn.c_attn(layernorm_output)

        query, key, value = mixed_x_layer.split(self.layer.attn.split_size, dim=2)
        query = query.view(bsz, sqlen, self.layer.attn.num_heads, self.layer.attn.head_dim)
        key = key.view(bsz, sqlen, self.layer.attn.num_heads, self.layer.attn.head_dim)
        value = value.view(bsz, sqlen, self.layer.attn.num_heads, self.layer.attn.head_dim)

        rotary_pos_emb = [cos_pos[:, -sqlen:, :, :], sin_pos[:, -sqlen:, :, :]]
        query = apply_rotary_pos_emb(query, rotary_pos_emb)
        key = apply_rotary_pos_emb(key, rotary_pos_emb)

        return query, key, value

class QwenPostAttnBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, attn_out, residual):
        attn_output = self.layer.attn.c_proj(attn_out)
        layernorm_input = attn_output + residual
        residual = layernorm_input
        layernorm_output = self.layer.ln_2(layernorm_input)
        mlp_output = self.layer.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        return hidden_states

class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        return m_logits

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')

def convert_embedding_cache():
    model = Embedding()
    input_ids = torch.tensor([range(BATCH_SIZE)], dtype=torch.int32)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding_cache.pt')

def convert_pre_attn_block(sqlen, layer_id):
    model = QwenPreAttnBlock(layer_id)
    hidden_states = torch.randn((1, sqlen, HIDDEN_SIZE))
    position_ids = torch.tensor([range(sqlen)], dtype=torch.int32)
    torch.onnx.export(
        model, (hidden_states, position_ids),
        f'{folder}/pre_attn_block_dyn_seq_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids'],
        output_names=['query', 'key', 'value'],
        dynamic_axes= {
            'input_states':{1:'seq_length'},
            'position_ids':{1:'seq_length'}},
        do_constant_folding=True,
        opset_version=15)

def convert_post_attn_block(sqlen, layer_id):
    model = QwenPostAttnBlock(layer_id)
    attn_out = torch.randn((1, sqlen, HIDDEN_SIZE))
    residual = torch.randn((1, sqlen, HIDDEN_SIZE))
    torch.onnx.export(
        model, (attn_out, residual),
        f'{folder}/post_attn_block_dyn_seq_{layer_id}.onnx',
        verbose=False,
        input_names=['attn_out','residual'],
        output_names=['block_out'],
        dynamic_axes= {
            'attn_out':{1:'seq_length'},
            'residual':{1:'seq_length'}},
        do_constant_folding=True,
        opset_version=15)

def convert_lm_head(sqlen):
    model = LmHead()
    hidden_states = torch.randn(1, sqlen, HIDDEN_SIZE)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/{sqlen}_lm_head.pt')

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\nSquence length: {SEQ_LENGTH}\n')
print(f"convert_embedding & lm_head")
convert_embedding()
convert_lm_head(16)
convert_lm_head(SEQ_LENGTH)

for i in range(NUM_LAYERS):
    print(f"convert_block_{i}")
    convert_pre_attn_block(SEQ_LENGTH, i)
    convert_post_attn_block(SEQ_LENGTH, i)
