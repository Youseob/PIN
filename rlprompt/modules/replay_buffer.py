import abc
import numpy as np
import gzip
import pickle

from collections import defaultdict, deque
from typing import Dict

class ReplayPool(object):
    @abc.abstractmethod
    def add_sample(self, sample):
        pass
    
    @abc.abstractmethod
    def size(self):
        pass
    
    @abc.abstractmethod
    def add_sequence(self, sequence):
        pass
    
    @abc.abstractmethod
    def random_batch(self, batch_size):
        pass
    
class FlexibleReplayPool(ReplayPool):
    def __init__(self, max_size, fields_attrs , modify_rew=False):
        super(FlexibleReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {}
        self.fields_attrs = {}

        self.add_fields(fields_attrs)
        self.modify_rew = modify_rew
        
        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0

    @property
    def size(self):
        return self._size

    @property
    def field_names(self):
        return list(self.fields.keys())
    #
    def add_fields(self, fields_attrs):
        self.fields_attrs.update(fields_attrs)
        for field_name, field_attrs in fields_attrs.items():
            field_shape = (self._max_size, *field_attrs['shape'])
            initializer = field_attrs.get('initializer', np.zeros)
            self.fields[field_name] = initializer(
                field_shape, dtype=field_attrs['dtype'])
    #
    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)
        self._samples_since_save += count
    #
    def add_sample(self, sample):
        samples = {
            key: value[None, ...]
            for key, value in sample.items()
        }
        self.add_samples(samples)
    #
    def add_samples(self, samples):
        """
        e.g samples = {
            'sample_ids': (bs, max_seq_len),
            'rewards': (bs, max_seq_len),
            'valid': (bs, max_seq_len)
        }
        """
        field_names = list(samples.keys())
        # if isinstance(samples[field_names[0]], np.ndarray):
        num_samples = len(samples[field_names[0]])
            
        index = np.arange(self._pointer, self._pointer + num_samples) % self._max_size
        for field_name in self.field_names:         
            assert field_name in field_names
            values = samples[field_name]                  
            try:
                assert values.shape[0] == num_samples, f'value shape: {values.shape[0]}, expected: {num_samples}'
                if isinstance(values[0], dict):
                    values = np.stack([np.concatenate([
                                value[key]
                                for key in value.keys()
                            ], axis=-1) for value in values])
                self.fields[field_name][index] = values
            except Exception as e:
                import traceback
                traceback.print_exc(limit=10)
                print('[ DEBUG ] errors occurs: {}'.format(e))
        self._advance(num_samples)

    def restore_samples(self, samples):
        num_samples = samples[list(samples.keys())[0]].shape[0]
        index = np.arange(
            0, num_samples) % self._max_size
        for key, values in samples.items():
            assert key in self.field_names
            self.fields[key][index] = values

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, last_n), self._pointer
        ) % self._max_size
        return self.batch_by_indices(
            last_n_indices, field_name_filter=field_name_filter, **kwargs)

    def filter_fields(self, field_names, field_name_filter):
        if isinstance(field_name_filter, str):
            field_name_filter = [field_name_filter]

        if isinstance(field_name_filter, (list, tuple)):
            field_name_list = field_name_filter

            def filter_fn(field_name):
                return field_name in field_name_list

        else:
            filter_fn = field_name_filter

        filtered_field_names = [
            field_name for field_name in field_names
            if filter_fn(field_name)
        ]

        return filtered_field_names

    def batch_by_indices(self, indices, field_name_filter=None):
        if np.any(indices % self._max_size > self.size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        field_names = self.field_names
        if field_name_filter is not None:
            field_names = self.filter_fields(
                field_names, field_name_filter)

        return {
            field_name: self.fields[field_name][indices]
            for field_name in field_names
        }

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0

    def load_experience(self, experience_path):
        with gzip.open(experience_path, 'rb') as f:
            latest_samples = pickle.load(f)

        key = list(latest_samples.keys())[0]
        num_samples = latest_samples[key].shape[0]
        for field_name, data in latest_samples.items():
            assert data.shape[0] == num_samples, data.shape

        self.add_samples(latest_samples)
        self._samples_since_save = 0

    def return_all_samples(self):
        return {
            field_name: self.fields[field_name][:self.size]
            for field_name in self.field_names
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state['fields'] = {
            field_name: self.fields[field_name][:self.size]
            for field_name in self.field_names
        }

        return state

    def __setstate__(self, state):
        if state['_size'] < state['_max_size']:
            pad_size = state['_max_size'] - state['_size']
            for field_name in state['fields'].keys():
                field_shape = state['fields_attrs'][field_name]['shape']
                state['fields'][field_name] = np.concatenate((
                    state['fields'][field_name],
                    np.zeros((pad_size, *field_shape))
                ), axis=0)

        self.__dict__ = state
        
class SimpleReplayTokenSeqPool(FlexibleReplayPool):
    
    def __init__(self, max_seq_len, state_dim, *args, **kwargs):
        # dim of hidddem_state 
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim # embedding_dim

        fields = {
            'states': {
                'shape': (self.max_seq_len, state_dim),
                'dtype': 'float32'  
            },
            'input_ids': {
                'shape': (self.max_seq_len,),
                'dtype': 'long'
            },
            'reward':{
                'shape': (),
                'dtype': 'float32'
            },
            'sequence_lengths': {
                'shape': (),
                'dtype': 'long'
            },
        }

        super(SimpleReplayTokenSeqPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)
    
    def random_batch_for_initial(self, batch_size):
        # random_indices = self.random_indices(batch_size)
        valids = np.sum(self.fields['valid'], axis=1).squeeze(-1)[:self.size]
        first_ind = np.random.choice(np.arange(self.size), p=valids/np.sum(valids), size=(batch_size, ))
        second_ind = []
        for ind, item in enumerate(first_ind):
            second_ind.append(np.random.randint(valids[item]))
        indices = [(a, b) for a, b in zip(first_ind, second_ind)]
        return self.batch_by_double_index(
            indices)
    
    def batch_by_double_index(self, indices):
        batch = {}
        for field in self.field_names:
            shapes = self.fields[field].shape
            shapes = (len(indices), shapes[-1])
            data = np.zeros(shapes, dtype=np.float32)
            for ind, item in enumerate(indices):
                data[ind] = self.fields[field][item[0], item[1]]
            batch[field] = data
        return batch
            
    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)
    
    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)


if __name__=="__main__":
    buffer = SimpleReplayTokenSeqPool(max_size=20, max_seq_len=5, state_dim=10)
    buffer.add_samples({
        'input_ids': np.random.randint(100, size=(16, 5)),
        'states': np.random.rand(16, 5, 10),
        'reward': np.random.rand(16),
        'sequence_lengths': np.ones((16, 5)),
    })
    print(buffer)
    import pdb; pdb.set_trace()
    batch = buffer.random_batch(batch_size=2)
    # List[int]
    print(batch)