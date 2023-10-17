import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

from flash_attn.flash_attn_interface import ParallelAttention1


def main():
    device = torch.device('cuda:0')

    batch_size = 1
    seq_len = 8
    hidden_size = 16
    num_attention_heads = 2

    para_attn1 = ParallelAttention1(hidden_size, num_attention_heads)
    para_attn1.to(device)
    para_attn1.half()

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device, requires_grad=True)
    dense_out = para_attn1(hidden_states)

    # print(f'hidden_states shape {hidden_states.shape}')
    # print(f'dense_out shape {dense_out.shape}')

    out = torch.sum(dense_out)
    out.backward()

    print('successful !!! ')

if __name__ == '__main__':
    main()