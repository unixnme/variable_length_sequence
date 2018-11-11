import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn


a = torch.Tensor([[1], [2], [3], [0]])
b = torch.Tensor([[4], [5]])
c = torch.Tensor([[6]])
packed = rnn_utils.pack_sequence([a, b, c])
print(packed)

lstm = nn.LSTM(1,3)

packed_output, (h,c) = lstm(packed)

y = rnn_utils.pad_packed_sequence(packed_output)
print(y)

seq = rnn_utils.PackedSequence()
