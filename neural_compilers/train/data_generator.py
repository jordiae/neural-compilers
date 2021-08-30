import logging
import json
import git
import ntpath
import time
import datetime
import uuid
from dirhash import dirhash
from dataclasses import asdict
from typing import Dict
from pathlib import Path
import os
from neural_compilers.utils.config import DataGenConfig
from neural_compilers.utils.tokenization import GCC, CodeTokenizer, SubwordTokenizer, Compiler
from neural_compilers.utils.utilities import run_command, timeit
from typing import Optional
from tqdm import tqdm
import shutil
import multiprocessing as mp

JOBS = 8


class DataGen:
    def __init__(self, config: DataGenConfig):
        self._config = config
        self._code_tokenizer: Optional[CodeTokenizer] = None
        self._subword_tokenizer: Optional[SubwordTokenizer] = None
        self._compiler: Optional[Compiler] = None
        self._output_dir = None
        self.is_setup = False

    def setup(self):
        assert not self.is_setup
        timestamp = time.strftime("%Y-%m-%d-%H%M")
        output_path = self._config.output_path
        input_name = ntpath.basename(self._config.input_path)
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        extra_id = uuid.uuid4().hex
        input_hash = dirhash(self._config.input_path, 'md5', jobs=JOBS)
        supervised = 'supervised' if self._config.supervised else 'unsupervised'
        output_dir = os.path.join(output_path,
                                  f'{input_name}-{supervised}-{timestamp}-{input_hash}-{sha[:4]}-'
                                  f'{extra_id[:4]}')
        self._output_dir = output_dir
        os.makedirs(output_dir)
        logging.basicConfig(filename=os.path.join(output_dir, 'gen-data.log'), level=logging.INFO)
        logging.getLogger('').addHandler(logging.StreamHandler())
        logging.info('hey')
        with open(os.path.join(output_dir, 'gen-data-config.json'), 'w') as f:
            json.dump(asdict(self._config), f, indent=4)

        self._code_tokenizer = CodeTokenizer.from_config(self._config.tokenizer_config)
        self._subword_tokenizer = SubwordTokenizer.from_config(self._config.tokenizer_config,
                                                               output_dir=self._output_dir)
        self._compiler = GCC()
        self.is_setup = True

    def _compile(self, raw_c: str) -> str:
        return self._compiler.compile(raw_c)

    def generate(self):
        t0 = datetime.datetime.now().timestamp()
        self.setup()
        stats = self._process_dataset(output_dir=self._output_dir)
        with open(os.path.join(self._output_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        t1 = datetime.datetime.now().timestamp()
        logging.info(f'Total: Elapsed {t1 - t0}s')

    def _process_file(self, path):
        # Read C, compile to assembly, discard (if required), and collect stats
        with open(path, 'r') as c_file:
            raw_c = c_file.read()
        raw_asm = self._compile(raw_c)  # self._compile(path, output_dir=output_dir)
        if len((raw_asm.split())) == 0:
            print('WARNING: Skipping empty assembly (cmpile error?)')
            return None
        # Pretokenize
        pretok_c = self._code_tokenizer.tokenize(raw_c, lang='c', just_func=self._config.just_func)
        pretok_asm = self._code_tokenizer.tokenize(raw_asm, lang='asm', just_func=self._config.just_func)
        total_len = len(pretok_c) + len(pretok_asm)
        # Note that we are discarding by length BEFORE subword tokenization.
        # It is neither correct or incorrect, but a rather arbitrary design choice
        if len(pretok_asm) < self._config.min_tokens // 2:
            return None
        elif total_len < self._config.min_tokens:
            return None
        elif total_len > self._config.max_tokens:
            return None
        else:
            c_path = os.path.join(self._output_dir, path)
            with open(c_path, 'w') as f:
                f.write(' '.join(pretok_c) + '\n')
            asm_path = c_path[:-1] + 's'
            with open(asm_path, 'w') as f:
                f.write(' '.join(pretok_asm) + '\n')
            return c_path

    def _process_dataset(self, output_dir: str) -> Dict:
        stats = {'c_length': 0, 'asm_length': 0, 'discarded_min': 0, 'discarded_max': 0, 'kept': 0}
        kept_files = []
        c_corpus_path = os.path.join(output_dir, 'corpus.c')
        c_corpus_subword = os.path.join(output_dir, 'corpus.tok.c')
        asm_corpus_path = os.path.join(output_dir, 'corpus.s')
        asm_corpus_subword = os.path.join(output_dir, 'corpus.tok.s')


        def ig_f(dir, files):
            return [f for f in files if os.path.isfile(os.path.join(dir, f))]

        mirrored_dir = os.path.join(self._output_dir, os.path.basename(self._config.input_path))
        shutil.copytree(self._config.input_path, mirrored_dir, ignore=ig_f)


        @timeit
        def read_tokenize_discard():

            tasks = list(Path(self._config.input_path).rglob('*.c'))#[:10000]
            with mp.Pool() as p:
                for _ in tqdm(p.imap_unordered(self._process_file, tasks), total=len(tasks)):
                   pass

        read_tokenize_discard()

        # cat
        # TODO: here we cannot use run_command because we are piping (fix)
        command = "find " + mirrored_dir + "  -iname '*.c' -print0 | sort -zn | xargs -0 -I '{}' cat '{}' - > " + os.path.join(self._output_dir, 'corpus.c')
        os.system(command)
        print(command)
        command = "find " + mirrored_dir + "  -iname '*.s' -print0 | sort -zn | xargs -0 -I '{}' cat '{}' - > " + os.path.join(
            self._output_dir, 'corpus.s')
        os.system(command)
        print(command)


        # Train-valid-test split

        @timeit
        def train_valid_test_split():
            # np.random.shuffle(kept_files)
            os.system(f"shuf {os.path.join(self._output_dir, 'corpus.c')} -o {os.path.join(self._output_dir, 'corpus.shuf.c')} --random-source={os.path.join(self._output_dir, 'corpus.c')}")
            os.system(f"shuf {os.path.join(self._output_dir, 'corpus.s')} -o {os.path.join(self._output_dir, 'corpus.shuf.s')} --random-source={os.path.join(self._output_dir, 'corpus.c')}")
            with open(os.path.join(self._output_dir, 'corpus.shuf.c'), 'r') as c, open(os.path.join(self._output_dir, 'corpus.shuf.s'), 'r') as s:
                lines = c.readlines()
                with open(os.path.join(self._output_dir, 'test.c'), 'w') as t:
                    t.writelines(lines[:self._config.valid_test_size])
                with open(os.path.join(self._output_dir, 'valid.c'), 'w') as t:
                    t.writelines(lines[self._config.valid_test_size:self._config.valid_test_size*2])
                with open(os.path.join(self._output_dir, 'train.c'), 'w') as t:
                    if self._config.max_train_data:
                        train_size = len(lines) - self._config.valid_test_size * 2
                        new_train_size = self._config.max_train_data
                        assert train_size >= new_train_size
                        diff = train_size - new_train_size
                        max_idx = len(lines) - diff
                        t.writelines(lines[self._config.valid_test_size * 2:max_idx])
                    else:
                        t.writelines(lines[self._config.valid_test_size*2:])

                lines = s.readlines()
                with open(os.path.join(self._output_dir, 'test.s'), 'w') as t:
                    t.writelines(lines[:self._config.valid_test_size])
                with open(os.path.join(self._output_dir, 'valid.s'), 'w') as t:
                    t.writelines(lines[self._config.valid_test_size:self._config.valid_test_size * 2])
                with open(os.path.join(self._output_dir, 'train.s'), 'w') as t:
                    if self._config.max_train_data:
                        t.writelines(lines[self._config.valid_test_size * 2:max_idx])
                    else:
                        t.writelines(lines[self._config.valid_test_size * 2:])


        train_valid_test_split()


        # Learn subword tokenizer
        # joint
        self._subword_tokenizer.train([os.path.join(self._output_dir, 'train.c'), os.path.join(self._output_dir, 'train.s')])


        # Apply subword tokenizer + write tokenized (text files à la Fairseq)
        @timeit
        def apply_tokenizer():
            for subset in ['train.c', 'valid.c', 'test.c', 'train.s', 'valid.s', 'test.s']:
                path = os.path.join(self._output_dir, subset)
                tok_path = os.path.join(self._output_dir, subset.split('.')[0] + '.tok.' + subset.split('.')[1])
                with open(tok_path, 'w') as tok_file:
                    with open(path, 'r') as pretok_file:
                        for line in pretok_file:
                            tokenized = ' '.join(self._subword_tokenizer.tokenize_program(line))
                            tok_file.write(tokenized + '\n')

        apply_tokenizer()

        # Fairseq preprocessing
        src_lang = 'c'
        tgt_lang = 's'
        data_path = self._output_dir
        train_prefix = os.path.join(data_path, 'train.tok')
        valid_prefix = os.path.join(data_path, 'valid.tok')
        test_prefix = os.path.join(data_path, 'test.tok')
        destdir = data_path
        threshold_vocab = 0
        workers = JOBS
        dict_options = '--joined-dictionary'
        fairseq_preprocess_comand = f'''
        fairseq-preprocess \
        --source-lang {src_lang} --target-lang {tgt_lang} \
        --trainpref {train_prefix} --validpref {valid_prefix} --testpref {test_prefix} \
        --destdir {destdir} --thresholdtgt {threshold_vocab} --thresholdsrc {threshold_vocab} {dict_options} \
        --workers {workers}
        '''
        print(fairseq_preprocess_comand)
        stdout, stderr = run_command(fairseq_preprocess_comand)
        print(stdout)
        print(stderr)

        return stats
