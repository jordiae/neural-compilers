# https://discuss.huggingface.co/t/how-to-add-additional-custom-pre-tokenization-processing/1637
# https://github.com/huggingface/tokenizers/blob/b24a2fc1781d5da4e6ebcd3ecb5b91edffc0a05f/bindings/python/examples/custom_components.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pygments import lexers
import re
from typing import List, Union
from neural_compilers.utils.utilities import run_command
import multiprocessing as mp
from neural_compilers.utils.config import SubwordType, TokenizerConfig, TokenizerType
import logging
from neural_compilers.utils.utilities import timeit
from neural_compilers.utils.constants import NEWLINE_ESCAPE
from pygments.token import Token
from typing import Optional
from tokenizers.pre_tokenizers import WhitespaceSplit


class Compiler:
    def compile(self, c_text: str) -> str:
        raise NotImplementedError

    def preprocess(self, c_text: str) -> str:
        raise NotImplementedError


class GCC(Compiler):
    def __init__(self, print_stderr: bool = False):
        super().__init__()
        self._print_stderr = print_stderr

    def compile(self, c_text: str) -> str:
        # make gcc read from stdin and write to stdout
        stdout, stderr = run_command(f'gcc -S -O0 -x c -o /dev/stdout -', stdin=c_text)
        if self._print_stderr and stderr:
            print(stderr)
        return stdout

    def preprocess(self, c_text: str) -> str:
        # make gcc read from stdin and write to stdout
        stdout, stderr = run_command(f'gcc -E -x c -o /dev/stdout -', stdin=c_text)
        if stderr:
            pass  # logging.warning(stderr)
        return stdout


class CodeTokenizer:
    @classmethod
    def from_config(cls, config: TokenizerConfig):
        if config.tokenizer_type == TokenizerType.PYGMENTS:
            return PygmentsTokenizer()
        raise NotImplementedError(config.tokenizer_type)

    def _tokenize_c(self, program: str, just_func: bool = False) -> List[str]:
        raise NotImplementedError

    def _tokenize_asm(self, program: str, just_func: bool = False) -> List[str]:
        raise NotImplementedError

    def tokenize(self, programs: Union[str, List[str]], lang: str,
                 par: bool = False, just_func: bool = False) -> Union[List[str], List[List[str]]]:
        is_str = False
        if isinstance(programs, str):
            is_str = True
            programs = [programs]
        if lang == 'c':
            tokenize_func = self._tokenize_c
        elif lang == 'asm':
            tokenize_func = self._tokenize_asm
        else:
            raise ValueError(lang)
        if par:
            if just_func:
                raise NotImplementedError
            with mp.Pool() as pool:
                tokenized = pool.map(tokenize_func, programs)
        else:
            tokenized = list(map(lambda x: tokenize_func(x, just_func=just_func), programs))
        if is_str:
            tokenized = tokenized[0]
        return tokenized


class PygmentsTokenizer(CodeTokenizer):
    def __init__(self):
        super().__init__()
        self._c = lexers.get_lexer_by_name('c')
        self._asm = lexers.get_lexer_by_name('gas')
        self._c_compiler = GCC()

    def _tokenize_c(self, program: str, just_func: bool = False) -> List[str]:
        program = self._c_compiler.preprocess(program)

        # keep only the function, if required
        def is_header_of_implemented_func(s):
            return '(' in s and ';' not in s and 'while' not in s and 'if' not in s

        if just_func:
            lines = []
            inside_func = False
            for line in program.splitlines():
                if not inside_func and is_header_of_implemented_func(line):
                    inside_func = True
                if inside_func:
                    lines.append(line)
            program = '\n'.join(lines)

        tokenized = [token.strip() for token_type, token in self._c.get_tokens(program) if len(token.split()) > 0 and
                     token_type not in Token.Comment]

        return tokenized

    def _tokenize_asm(self, program: str, just_func: bool = False) -> List[str]:
        filtered_program = []
        # Filter metadata
        inside_implemented_func = False
        for line in program.splitlines():
            if line.strip().startswith('.file'):  # file name (metadata)
                continue
            if line.strip().startswith('.ident'):  # compiler/OS version (metadata):
                continue
            if not just_func:
                filtered_program.append(line)
            else:  # logic for just keeping the procedure
                if just_func and inside_implemented_func:
                    if '.cfi_endproc' in line:
                        filtered_program.append(line)
                        break


                pattern = re.compile("(_[A-Z]|[a-z])\w:")
                if not inside_implemented_func and pattern.search(line):
                    inside_implemented_func = True
                if inside_implemented_func:
                    filtered_program.append(line)

        filtered_program = '\n'.join(filtered_program)
        tokenized =\
            [token for token_type, token in self._asm.get_tokens(filtered_program) if token_type not in Token.Comment]
        # newline needed (end of statement)
        tokenized = [NEWLINE_ESCAPE if token == '\n' else token for token in tokenized]
        tokenized = (' '.join(tokenized)).split()
        return tokenized


class SubwordTokenizer:
    def __init__(self, subword_tokenizer_type: SubwordType, subword_vocab_size: int, shared_vocab: bool,
                 output_dir: str):
        if not shared_vocab:
            raise NotImplementedError('Not sharing vocab')
        self._vocab_size = subword_vocab_size
        self.type = subword_tokenizer_type
        self.trained = False
        self.hf_tokenizer: Optional[Tokenizer] = None
        self.output_dir = output_dir

    @classmethod
    def from_config(cls, config: TokenizerConfig, output_dir: str):
        return cls(config.subword_tokenizer, config.subword_vocab_size, config.shared_vocab, output_dir=output_dir)

    def restore(self, tok_path: str):
        assert not self.trained
        self.hf_tokenizer = Tokenizer.from_file(tok_path)
        self.trained = True

    @timeit
    def train(self, files: List[str]):
        assert not self.trained
        if self.type != 'subword-nmt':
            raise NotImplementedError(self.type)

        special_tokens = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<newline>"  # assembly
        ]
        import string
        ascii_alphabet = string.printable
        self.hf_tokenizer = Tokenizer(BPE())
        # Needed to restore (HF bug): https://github.com/huggingface/tokenizers/issues/566
        self.hf_tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=self._vocab_size, show_progress=True,
                             initial_alphabet=ascii_alphabet, min_frequency=0, continuing_subword_prefix='##')
        self.hf_tokenizer.train(files=files, trainer=trainer)
        self.hf_tokenizer.enable_truncation(max_length=512)
        logging.info(f"saving model tokenizer to {self.output_dir}")
        import os
        self.hf_tokenizer.save(os.path.join(self.output_dir, 'tokenizer.json'))
        self.trained = True

    def tokenize_program(self, program: str) -> List[str]:
        line = self.hf_tokenizer.encode(program, add_special_tokens=False).tokens  # no special tokens -> fairseq
        return line

    def tokenize_programs(self, programs: List[str]) -> List[List[str]]:
        tokenized = []
        for program in programs:
            tokenized.append(self.tokenize_program(program))
        return tokenized

