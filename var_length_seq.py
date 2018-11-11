import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data.dataloader import _use_shared_memory, re, numpy_type_map, int_classes, string_classes, DataLoader
import collections


class VariableLengthSequence(object):
    def __init__(self, seq):
        '''
        :param seq: sequence of torch seq
        '''
        self.padded_seq = rnn_utils.pad_sequence(seq, batch_first=True)
        self.lengths = [len(s) for s in seq]

    def __str__(self):
        return str(self.padded_seq) + '\n' + str(self.lengths)


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


class VariableLengthSequenceDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(VariableLengthSequenceDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                                               num_workers, default_collate, pin_memory, drop_last,
                                                               timeout, worker_init_fn)


if __name__ == '__main__':
    from dataset import RandomDataset
    from torch.utils.data import DataLoader

    dataset = RandomDataset()
    dataloader = VariableLengthSequenceDataLoader(dataset, 64, True)

    for x in dataloader:
        pass
