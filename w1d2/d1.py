# %%
#!cp -r /Users/tilman/Documents/projects/arena/arena-v1-ldn/w1d2 /Users/tilman/Documents/projects/arena/arena/w1d2

# %%
import torch as t
import torch.nn as nn
from fancy_einsum import einsum
import einops
import w1d2.utils as utils
import plotly.express as px
# %%


def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (batch, seq_len, emb_len)
    K: shape (batch, seq_len, emb_len)
    V: shape (batch, seq_len, emb_len)

    Return: shape (batch, seq_len, emb_len)
    '''

    emb_len = Q.shape[-1]
    QKT = einsum('b s_q h, b h s_k -> b s_q s_k',
                 Q, K.transpose(dim0=-1, dim1=-2))
    QKT = QKT / (emb_len ** 0.5)
    attention_probs = t.softmax(QKT, dim=-1)

    attention_values = einsum(
        'b s_q s_k, b s_k h -> b s_q h', attention_probs, V)
    #attention_values = einsum(' b i j, b k j-> b i k', attention_probs, V)
    return attention_values


b = 64
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, emb)), t.rand((b, seq, emb)), t.rand((b, seq, emb))

av = single_head_attention(Q, K, V)
assert av.shape == t.Size([64, 12, 8])

# %%
Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_attention(Q, K, V)
# px.imshow(Q.squeeze())
# px.imshow(K.squeeze())

px.imshow(av.squeeze())
#assert av.shape == t.Size([64, 12, 8])

# %%


def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (batch, seq_len, emb_len)
    K: shape (batch, seq_len, emb_len)
    V: shape (batch, seq_len, emb_len)

    Return: shape (batch, seq_len, emb_len)
    '''

    emb_len = Q.shape[-1]
    seq_len = Q.shape[-2]
    QKT = einsum('b s_q h, b h s_k -> b s_q s_k',
                 Q, K.transpose(dim0=-1, dim1=-2))
    QKT = QKT / (emb_len ** 0.5)
    tri = t.tril(t.ones((seq_len, seq_len)), diagonal=0)*(-10 ** 4)
    QKT_masked = (QKT - tri)
    attention_probs = t.softmax(QKT_masked, dim=-1)

    attention_values = einsum(
        'b s_q s_k, b s_k h -> b s_q h', attention_probs, V)
    #attention_values = einsum(' b i j, b k j-> b i k', attention_probs, V)
    return attention_values


# %%
Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_masked_attention(Q, K, V)
px.imshow(av.squeeze())
# %%
def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)

    Return: shape (batch, seq_len, nheads*headsize)
    '''
    
    emb_len = Q.shape[-1]
    seq_len = Q.shape[-2]
    headsize = emb_len // num_heads

    Q_ = einops.rearrange(Q, 'b s (nh h) -> b nh s h', nh=num_heads)
    K_ = einops.rearrange(K, 'b s (nh h) -> b nh s h', nh=num_heads)
    V_ = einops.rearrange(V, 'b s (nh h) -> b nh s h', nh=num_heads)

    QKT = einsum('b nh s_q h, b nh h s_k -> b nh s_q s_k',
                 Q_, K_.transpose(dim0=-1, dim1=-2))
    QKT = QKT / (emb_len ** 0.5)
    tri = t.tril(t.ones((seq_len, seq_len)), diagonal=0)*(-10 ** 4)
    QKT_masked = (QKT - tri)
    attention_probs = t.softmax(QKT_masked, dim=-1)

    attention_values_ = einsum(
        'b nh s_q s_k, b nh s_k h -> b nh s_q h', attention_probs, V_)
    # b hn s_q h -->e = n*h --> b s_q e
    attention_values = einops.rearrange(attention_values_, ' b hn s_q h ->  b s_q (hn h)')

    return attention_values

# %%
b = 64
n_h = 2
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, n_h*emb)), t.rand((b, seq, n_h*emb)), t.rand((b, seq, n_h*emb))

av = multihead_masked_attention(Q, K, V, num_heads=n_h)
assert av.shape == t.Size([64, 12, 16])


# %%
class MultiheadMaskedAttention(nn.Module):
    # Hydra-Attention 
    # W_QKV: nn.Linear 
    # W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_QKV = nn.Linear(hidden_size, 3*emb)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        # QKV = einsum('b s hs, hs h2 -> b (s h2) hs',x, self.W_QKV) # h2 = 3*emb_len 
        QKV = self.W_QKV(x)
        QKV_ = einops.rearrange(QKV, 'b hs (n es) -> n b hs es', n=3)
        Q, K, V = QKV_[0], QKV_[1], QKV_[2]
        av = multihead_masked_attention(Q, K, V, self.num_heads)
        # av shape: b s_q emb
        # W_O shape: emb, emb
        out = self.W_O(av)
        return out

# %%
b = 128
n_h = 4
seq = 16
emb = 48
hydra = MultiheadMaskedAttention(hidden_size=emb, num_heads=n_h)

x = t.rand((b, seq, emb))
assert hydra(x).shape == t.Size([128, 16, 48])



# %%












