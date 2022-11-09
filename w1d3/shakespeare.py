# %%
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Union
import re
from fancy_einsum import einsum
import einops
from dataclasses import dataclass
import plotly.express as px
from torchinfo import torchinfo
import numpy as np

import dev

device = t.device('cpu')
#%%
class WordsDataset(Dataset):
    def __init__(self, words, max_seq_len, fraction=0.1):
        self.words = words
        self.vocab_size = len(set(words))
        self.max_seq_len = max_seq_len
        self.fraction = fraction
        self.max_len = int(self.vocab_size - (self.max_seq_len + 1))

        # shamelessly stolen 
        # if we call set on words we reduce the corpus for words that appear double 
        # sorted (alphabetically) enumerated from 1 to dict_length 
        # we get a set of words with their corresponding number
        self.words_to_token_idx = {word: idx for (idx, word) in enumerate(sorted(set(words)))}
        
        # that dict but reversed, getting a set of numbers with their corresponding words
        self.token_idx_to_words = {idx: word for (word, idx) in self.words_to_token_idx.items()}
        assert len(self.token_idx_to_words) == (self.vocab_size)

        # this is our corpus of words in tokens 
        self.tokens = t.tensor([self.words_to_token_idx[word] for word in words]).to(dtype=t.long)

    def __len__(self):
        return int(self.max_len * self.fraction)

    def __getitem__(self, idx):
        
        next_token = self.tokens[idx + self.seq_len: idx + self.seq_len + 1] # last token of sequenth length
        tokens = self.tokens[idx: idx + self.seq_len] # all token before that 
        
        return tokens, next_token

with open('w1d3/100-0.txt', encoding="utf-8") as f:
    lines = f.read()
    words = re.split(r"\b", lines)

max_sequence_length = 32
fraction_of_dataset = 0.1
dataset = WordsDataset(words=words, max_seq_len=max_sequence_length, fraction=fraction_of_dataset)

# %%

# %%
class WordsTokenizer():
    def __init__(self, wordsdataset: WordsDataset):
        self.words_to_token_idx = wordsdataset.words_to_token_idx
        self.token_idx_to_words = wordsdataset.token_idx_to_words
        self.model_max_length = 5
    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        split_text = re.split(r"\b", initial_text)
        token_list = []
        for t in split_text:
            if len(t)>0:
                token_list.append( self.words_to_token_idx[t] )
        if return_tensors is None:
            return token_list
        elif return_tensors == 'pt':
            return t.tensor(token_list)
        else:
            raise Exception("Invalid return_tensor, either _pt_ or None")

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        sentence = ''
        for t in list_of_ids:
            sentence += '' + str(self.words_to_token_idx[int(t)] )
        return sentence
# %%


# %%
initial_text = "turn down for what"

config = dev.TransformerConfig(
    num_layers=2,
    num_heads=4,
    vocab_size=dataset.vocab_size,
    hidden_size=128,
    max_seq_len=5,
    dropout=0.1,
    layer_norm_epsilon=1e-5,
    
)

trainset = WordsDataset(words=words, max_seq_len=max_sequence_length, fraction=fraction_of_dataset)

model = dev.DecoderOnlyTransformer(config).to(device).train()

tokenizer = WordsTokenizer(trainset)

text_output = dev.sample_tokens(model, tokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)

print(text_output)



# %%
batch_size=int(32)

model = dev.DecoderOnlyTransformer(config).to(device).train()

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)
dev.train(model, optimizer, train_loader, test_loader, criterion, num_epochs=10, device=device)




# %%


# %%




# %%




# %%




# %%




# %%




# %%




# %%




# %%




# %%




# %%




# %%



# %%




# %%




