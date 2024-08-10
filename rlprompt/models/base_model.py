from torch import nn

SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl',
                 'EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B',
                 'facebook/opt-125m', 'facebook/opt-350m', 
                 'facebook/opt-1.3b', 'facebook/opt-2.7b', 
                #  'facebook/opt-6.7b', 'facebook/opt-13b' 
                 ]

LM_HIDDEN_SIZES = {'distilgpt2': 768,
                   'gpt2': 768,
                   'gpt2-medium': 1024,
                   'gpt2-large': 1280,
                   'gpt2-xl': 1600,
                   'EleutherAI/gpt-neo-125m': 768,
                   'EleutherAI/gpt-neo-1.3B': 2048,
                   'facebook/opt-125m': 768,
                   'facebook/opt-350m': 512,
                   'facebook/opt-1.3b': 2048,
                #    'facebook/opt-2.7b': 
                   }
BOS_TOKENS = {
    'distilgpt2': '<|endoftext|>',
    'gpt2': '<|endoftext|>',
    'gpt2-medium': '<|endoftext|>',
    'gpt2-large': '<|endoftext|>',
    'gpt2-xl': '<|endoftext|>',
    'EleutherAI/gpt-neo-125m': '<|endoftext|>',
    'EleutherAI/gpt-neo-1.3B': '<|endoftext|>',
    'facebook/opt-125m': '</s>',
    'facebook/opt-350m': '</s>',
    'facebook/opt-1.3b': '</s>',    
}

PAD_TOKENS = {
    'distilgpt2': '<|endoftext|>',
    'gpt2': '<|endoftext|>',
    'gpt2-medium': '<|endoftext|>',
    'gpt2-large': '<|endoftext|>',
    'gpt2-xl': '<|endoftext|>',
    'EleutherAI/gpt-neo-125m': '<|endoftext|>',
    'EleutherAI/gpt-neo-1.3B': '<|endoftext|>',
    'facebook/opt-125m': '<pad>',
    'facebook/opt-350m': '<pad>',
    'facebook/opt-1.3b': '<pad>',    
}
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def greedy_search(self, *args, **kwargs):
        raise NotImplementedError

    def teacher_forcing(self, *args, **kwargs):
        raise NotImplementedError
