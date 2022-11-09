# %%
#!cp -r /Users/tilman/Documents/projects/arena/arena-v1-ldn/w1d3 /Users/tilman/Documents/projects/arena/arena/w1d3

# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from einops import rearrange
import os
import wandb
from torch.utils.data import DataLoader
import time
import torch as t
import torch.nn as nn
from fancy_einsum import einsum
import einops
from dataclasses import dataclass
import plotly.express as px
from torchinfo import torchinfo
from torch.utils.data import Dataset
import numpy as np

#device = t.device("mps" if t.backends.mps.is_available() else "cpu")
device = t.device("cpu")
# %%


class RevSequenceDataset(Dataset):
    def __init__(self, seq_length=5, data_set_size=10000):
        self.data_set_size = data_set_size
        self.seq_length = seq_length

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, idx):
        sequence = t.tensor(np.random.choice(
            10, self.seq_length, replace=False))
        rev_sequence = t.flip(sequence, dims=(0,))
        return (sequence, rev_sequence)


class CustomTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    @staticmethod
    def from_config(config, samples):
        texts = [t.randint(high=config.vocab_size, size=(
            config.max_seq_len,)) for _ in range(samples)]
        labels = [t.flip(text, (0,)) for text in texts]
        return CustomTextDataset(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample


# %%
class MLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.linear_1 = nn.Linear(self.hidden_size, 4*self.hidden_size)
        self.linear_2 = nn.Linear(4*self.hidden_size, self.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.gelu(self.linear_1(x))
        x = self.dropout(self.linear_2(x))
        return x


# %%

class PositionalEncoding(nn.Module):

    def __init__(self, max_sequence_length: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        self.maximal_pe = self.get_maximal_pe()
        #self.d = 10000

    def get_maximal_pe(self):

        def PE(delta):
            hidden_dim = self.hidden_dim

            sin_vec = t.sin(
                delta / 10000**(2 * t.arange(hidden_dim // 2) / hidden_dim))
            cos_vec = t.cos(
                delta / 10000**(2 * t.arange(hidden_dim // 2) / hidden_dim))

            pe = t.zeros(hidden_dim)
            pe[::2] = sin_vec
            pe[1::2] = cos_vec

            return pe

        pe = t.stack([PE(i) for i in range(self.max_sequence_length)])
        return pe

    def forward(self, x):
        '''
        x has shape (n, seq_len, hidden_dim)
        '''
        device = x.device
        return x + self.maximal_pe[:x.size(1), :].to(device)


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


def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor,
                               num_heads: int):
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

    QKT = einsum('b nh s_q h, b nh s_k h -> b nh s_q s_k', Q_, K_)
    QKT = QKT / (headsize**0.5)
    tri = t.triu(t.ones((seq_len, seq_len)), diagonal=1)*(-10 ** 4)
    QKT_masked = (QKT + tri)
    attention_probs = t.softmax(QKT_masked, dim=-1)

    attention_values_ = einsum('b nh s_q s_k, b nh s_k h -> b nh s_q h',
                               attention_probs, V_)
    # b hn s_q h -->e = n*h --> b s_q e
    attention_values = einops.rearrange(attention_values_,
                                        ' b hn s_q h ->  b s_q (hn h)')

    return attention_values


class MultiheadMaskedAttention(nn.Module):
    # Hydra-Attention
    # W_QKV: nn.Linear
    # W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert (hidden_size % num_heads) == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        Return: shape (batch, seq, hidden_size)
        '''
        # QKV = einsum('b s hs, hs h2 -> b (s h2) hs',x, self.W_QKV) # h2 = 3*emb_len
        QKV = self.W_QKV(x)
        Q, K, V = einops.rearrange(QKV, 'b hs (n es) -> n b hs es', n=3)
        av = multihead_masked_attention(Q, K, V, self.num_heads)
        # av shape: b s_q emb
        # W_O shape: emb, embs
        out = self.W_O(av)
        return out


@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.MLP = MLP(self.config.hidden_size, self.config.dropout)
        self.multiheaded_self_attention = MultiheadMaskedAttention(
            config.hidden_size, config.num_heads)
        self.layer_norm_1 = nn.LayerNorm(
            self.config.hidden_size, eps=config.layer_norm_epsilon)
        self.layer_norm_2 = nn.LayerNorm(
            self.config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:

        x = x + self.layer_norm_1(self.multiheaded_self_attention(x))
        x = x + self.layer_norm_2(self.MLP(x))

        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, embedding_dim=config.hidden_size)
        self.positional_embedding = PositionalEncoding(
            config.max_seq_len, hidden_dim=config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm_final = nn.LayerNorm(
            config.hidden_size, config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:

        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        x = self.decoder_blocks(x)
        x = self.layer_norm_final(x)

        x = einsum('word emb, b seq emb -> b seq word',
                   self.token_embedding.weight, x)

        return x


# %%
b = 64
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, emb)), t.rand((b, seq, emb)), t.rand((b, seq, emb))

av = single_head_attention(Q, K, V)
assert av.shape == t.Size([b, seq, emb])


Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_masked_attention(Q, K, V)
# px.imshow(av.squeeze()).show()

b = 64
n_h = 2
seq = 12
emb = 8
hydra = MultiheadMaskedAttention(hidden_size=emb, num_heads=n_h)

x = t.rand((b, seq, emb))

# %%

WANDB = 0

batch_size = 32
transformer_config = TransformerConfig(
    num_layers=2,
    num_heads=4,
    vocab_size=10,
    hidden_size=128,
    max_seq_len=5,
    dropout=0.1,
    layer_norm_epsilon=1e-5
)
model = DecoderOnlyTransformer(config=transformer_config)

if WANDB:
    wandb.init(project='RevSeqTransformer')
    wandb.config = transformer_config


def train(model, optimizer, trainloader, testloader, criterion, num_epochs=10, device=device):

    since = time.time()

    print("Beginning Training")
    data = next(iter(testloader))
    example_output = model(data[0])[:3]
    print("expected results")
    print(data[1][:3])
    print("current, bad, output")
    print(t.argmax(example_output, dim=-1))
    print("="*30)

    for epoch in range(num_epochs):

        model.to(device)
        model.train()

        running_loss = 0.0
        training_loss = 0.0
        progress_bar = tqdm(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds_rearranged = einops.rearrange(preds, "b s v -> (b s) v")

            y_rearranged = einops.rearrange(y, "b s -> (b s)")

            training_loss = criterion(preds_rearranged, y_rearranged)

            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += training_loss.item() * x.size(0)  # scale to n in batch
            progress_bar.set_description(
                "Epoch {} Training Loss {:.4f}".format(epoch, training_loss))
            if WANDB:
                wandb.log({"training_loss": training_loss.item()})

        epoch_loss = running_loss / len(trainloader.dataset)
        print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))

        data = next(iter(testloader))
        example_output = model(data[0].to(device))[:3]
        print(data[1][:3])
        print(t.argmax(example_output, dim=-1))
        print("="*30)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    return model


dataset = RevSequenceDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)

if False:
    model = train(model, optimizer,  train_loader, test_loader,
              criterion, num_epochs=10, device=device)

# %%
torchinfo.summary(model, input_data=t.tensor([[1, 2, 3, 4, 5]]))

# %%
#!pip install - q git+https: // github.com/huggingface/transformers.git

# %%

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id)

# %%
# encode context the generation is conditioned on
input_ids = tokenizer.encode(
    'I enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

# print("Output:\n" + 50 * '-')
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# %%
# activate beam search and early_stopping
beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

# print("Output:\n" + 50 * '-')
# print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %%
# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids,
    max_length=500,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# print("Output:\n" + 50 * '-')
# print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# %%
# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)

# now we have 3 output sequences
# print("Output:\n" + 50 * '-')
# for i, beam_output in enumerate(beam_outputs):
#     print("{}: {}".format(i, tokenizer.decode(
#         beam_output, skip_special_tokens=True)))
# %%
# set seed to reproduce results. Feel free to change the seed though to get different results
t.manual_seed(0)
# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=200,
    top_k=50,
    # set temperatur
    # temperature=0.6,
    top_p=0.9,
)

# print("Output:\n" + 50 * '-')
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# %%
import torch as t
import torch.nn.functional as F
import transformers

# gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# %%

def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k !=
                0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(
            input_ids + generated, dtype=t.int64, device=device)
        new_input_ids_truncated = new_input_ids[-min(
            tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    next_token = t.argmax(logits)

    return next_token 

# prompt = "Jingle bells, jingle bells, jingle all the way"
# print("Greedy decoding with prompt: ", prompt)
# output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
# print(f"Your model said: {output}")
# expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
# assert output == expected

# print("Greedy decoding a second time (should be deterministic): ")
# output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
# print(f"Your model said: {output}")
# expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
# assert output == expected

# print("Tests passed!")

# %%
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    sampled_token = t.distributions.categorical.Categorical(logits=logits)
    return sampled_token.sample()

# N = 20000
# probs = t.linspace(0, 0.4, 5)
# unnormalized_logits = probs.log() + 1.2345
# samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
# counts = t.bincount(samples, minlength=len(probs)) / N
# print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
# t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
# print("Tests passed!")

# %%
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    logits = logits / temperature
    return logits 

# logits = t.tensor([1, 2]).log()
# cold_logits = apply_temperature(logits, 0.001)
# print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
# t.testing.assert_close(cold_logits, 1000.0 * logits)
# hot_logits = apply_temperature(logits, 1000.0)
# print("A high temperature flattens the distribution: ", hot_logits)
# t.testing.assert_close(hot_logits, 0.001 * logits)
# print("Tests passed!")

# %%
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    count = t.bincount(input_ids)
    for idx, i in enumerate(count):
        # Probably very inefficient: do better.
        if i>1:
            logits[idx] = logits[idx] - i * freq_penalty

    return logits 

# bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
# input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
# logits = t.ones(tokenizer.vocab_size)
# penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
# assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
# assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
# print("Tests passed!")

# %%
# N_RUNS = 1
# your_prompt = "Jingle bells, jingle bells, jingle all the way"
# cases = [
#     ("High freq penalty", dict(freq_penalty=100.0)),
#     ("Negative freq penalty", dict(freq_penalty=-1.0)),
#     ("Too hot!", dict(temperature=2.0)),
#     ("Pleasantly cool", dict(temperature=0.7)),
#     ("Pleasantly warm", dict(temperature=0.9)),
#     ("Too cold!", dict(temperature=0.01)),
# ]
# for (name, kwargs) in cases:
#     for i in range(N_RUNS):
#         output = sample_tokens(gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
#         print(f"Sample {i} with: {name} ({kwargs}):")
#         print(f"Your model said: {repr(output)}\n")

# %%
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    top, top_idx = t.topk(logits, top_k)
    
    sampled_token = t.distributions.categorical.Categorical(logits=top)

    return top_idx[sampled_token.sample().item()] 

# k = 3
# probs = t.linspace(0, 0.4, 5)
# unnormalized_logits = probs.log() + 1.2345
# samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
# counts = t.bincount(samples, minlength=len(probs)) / N
# expected = probs.clone()
# expected[:-k] = 0
# expected /= expected.sum()
# print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
# t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
# print("Tests passed!")

# # %%
# your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
# output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
# print(f"Your model said: {repr(output)}")

# %%
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    sorted, indices = t.sort(logits, descending=True)
    cum_probs = sorted.softmax(-1).cumsum(-1)
    sorted_cum_probs =t.searchsorted(cum_probs, top_p, side="right").item() + 1
    if sorted_cum_probs < min_tokens_to_keep:
        sorted_cum_probs = min_tokens_to_keep
    idx = indices[:sorted_cum_probs]
    keep_logits = logits[idx]
    sample_token = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    
    return idx[sample_token].item()

# N = 2000
# unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
# samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
# counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
# print("top_p of 0.5 or lower should only return token 2: ", counts)
# assert counts[0] == 0 and counts[1] == 0

# N = 2000
# unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
# samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
# counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
# print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
# assert counts[0] == 0

# N = 4000
# top_p = 0.71
# probs = t.linspace(0, 0.4, 5)
# unnormalized_logits = probs.log() + 1.2345
# samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
# counts = t.bincount(samples, minlength=len(probs)) / N
# expected = probs.clone()
# expected[0:2] = 0
# expected /= expected.sum()
# print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
# t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

# print("All tests passed!")

# %%

# %%


# %%

# %%

# %%