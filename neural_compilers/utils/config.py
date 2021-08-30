from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TokenizerType(str, Enum):
    PYGMENTS = 'pygments'


class SubwordType(str, Enum):
    SUBWORD_NMT = 'subword-nmt'


@dataclass
class TokenizerConfig:
    tokenizer_type: TokenizerType
    subword_tokenizer: SubwordType
    subword_vocab_size: int
    shared_vocab: bool


@dataclass
class DataGenConfig:
    input_path: str
    output_path: str
    min_tokens: int
    max_tokens: int
    supervised: bool
    valid_test_size: int
    seed: int
    tokenizer_config: TokenizerConfig
    just_func: bool = False
    config_path: Optional[str] = None
    max_train_data: Optional[int] = None

    @classmethod
    def from_dict(cls, d):
        res = cls(**d)
        res.tokenizer_config = TokenizerConfig(**d['tokenizer_config'])
        return res
