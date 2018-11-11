import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data.dataloader import _use_shared_memory, re, numpy_type_map, int_classes, string_classes, DataLoader
import collections
import torch.nn as nn

class VariableLengthSequence(object):
    def __init__(self, seq, lengths=None):
        '''
        :param seq: sequence of torch seq
        '''
        if isinstance(seq, (list, tuple)):
            self.padded_seq = rnn_utils.pad_sequence(seq, batch_first=True)
            self.lengths = torch.LongTensor([len(s) for s in seq])
        elif isinstance(seq, torch.Tensor):
            self.padded_seq = seq.clone()
            self.lengths = lengths.clone()

    def __str__(self):
        return str(self.padded_seq) + '\n' + str(self.lengths)

    def __len__(self):
        return len(self.lengths)

    def seq(self):
        result = []
        for s,l in zip(self.padded_seq, self.lengths):
            result.append(s[...,:l])
        return result

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            try:
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            except:
                return VariableLengthSequence([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class VariableLengthDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(VariableLengthDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                                       num_workers, default_collate, pin_memory, drop_last,
                                                       timeout, worker_init_fn)

class VariableLengthConv1d(nn.Conv1d):
    def forward(self, input):
        if isinstance(input, torch.Tensor):
            x = input
        elif isinstance(input, VariableLengthSequence):
            x = input.padded_seq
            lengths = self.output_length([input.lengths])[0]
        x = super(VariableLengthConv1d, self).forward(x)
        if isinstance(input, VariableLengthSequence):
            x = VariableLengthSequence(x, lengths)
        return x

    def output_length(self, input_lengths):
        result = []
        for input_length, padding, dilation, kernel_size, stride in zip(input_lengths, self.padding, self.dilation, self.kernel_size, self.stride):
            num = input_length.type(torch.float32) + 2*padding - dilation * (kernel_size - 1) - 1
            denom = stride
            result.append(torch.floor(num / denom + 1))
        return torch.stack(result).type(torch.long)

if __name__ == '__main__':
    from dataset import RandomDataset
    from torch.utils.data import DataLoader

    dataset = RandomDataset()
    dataloader = VariableLengthDataLoader(dataset, 64, True)

    conv = VariableLengthConv1d(1, 1, 3, stride=1, padding=1)
    for x in dataloader:
        x.padded_seq = x.padded_seq.unsqueeze(1)
        y = conv(x)

        for idx,x_ in enumerate(x.seq()):
            y_ = conv(x_.unsqueeze(1))
            length = y.lengths[idx]
            assert torch.all(y_ == y.padded_seq[idx,...,:length])