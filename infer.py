import logging
from pathlib import Path
import os
from neural_compilers.utils.config import TokenizerConfig
from neural_compilers.utils.tokenization import CodeTokenizer, SubwordTokenizer
import json
import git
import time
from neural_compilers.utils.utilities import run_command
import uuid
from neural_compilers.utils.utilities import init_logging
from typing import List
from typing import Tuple
import shutil
from neural_compilers.utils.utilities import get_tmp_file
import logging
from copy import deepcopy


def tokenize(raw_code: str, subword_tokenizer: SubwordTokenizer, code_tokenizer: CodeTokenizer,
             replace_true_false: bool) -> str:
    c_code = raw_code
    # hack to avoid includes: TODO: review
    new_c_code = ''
    for line in c_code.splitlines():
        if not line.startswith('#'):
            new_c_code += line
    if 'stdbool' in raw_code and replace_true_false:
        new_c_code = raw_code.replace('true', '1').replace('false', '0')
    c_code = new_c_code
    pretok_c = ' '.join(code_tokenizer.tokenize(c_code, lang='c'))
    # subword tokenization
    tok_c = ' '.join(subword_tokenizer.tokenize_program(pretok_c))
    return tok_c


def cat_tokenize(c_path: str, model_path: str, eval_path: str, replace_true_false: bool) -> Tuple[str, List[str]]:
    with open(os.path.join(model_path, 'gen-data-config.json'), 'r') as f:
        tok_config_dict = json.load(f)['tokenizer_config']
    tok_config = TokenizerConfig(**tok_config_dict)
    code_tokenizer = CodeTokenizer.from_config(tok_config)
    tokenizer = SubwordTokenizer.from_config(tok_config, output_dir='__')
    tokenizer.restore(os.path.join(model_path, 'tokenizer.json'))
    cat_path = os.path.join(eval_path, 'all.c')
    c_files = sorted(Path(c_path).rglob('*.c'))
    if os.path.isdir(c_path):
        with open(cat_path, 'w') as outfile:
            for c_file in c_files:
                with open(c_file, 'r') as infile:
                    outfile.write(tokenize(infile.read(), subword_tokenizer=tokenizer, code_tokenizer=code_tokenizer,
                                           replace_true_false=replace_true_false)
                                  + '\n')
        return cat_path, list(map(str, c_files))
    else:
        with open(cat_path, 'w') as outfile:
            with open(c_path, 'r') as infile:
                for line in infile.readlines():
                    outfile.write(tokenize(line, subword_tokenizer=tokenizer, code_tokenizer=code_tokenizer,
                                           replace_true_false=replace_true_false))
        return cat_path, [c_path]


def run_model(data_path: str, model_path: str, beam: int, top_n: int, checkpoint_path: str) -> List[List[str]]:
    directions = '-s c -t s'
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(model_path, checkpoint_path)
    command = f'fairseq-interactive {model_path} {directions} --path {checkpoint_path} --beam {beam} --nbest {top_n}'

    with open(data_path, 'r') as f:
        c_code = f.read()
    stdout, stderr = run_command(command, stdin=c_code)
    res = []
    current = []  # potentially multiple hypotheses
    for l in stdout.splitlines():
        if l.startswith('H'):
            detokenized = l.split('\t')[2].replace(' ##', '')
            current.append(detokenized)
        elif l.startswith('S') and len(current) > 0:
            res.append(current)
            current = []
    if len(current) > 0:
        res.append(current)
    if len(res) == 0:
        print('ERROR')
        logging.error(stderr)
        exit()
    return res


def get_asm_header_and_footer(c_path: str) -> Tuple[str, str]:
    # first compile with gcc to get header and footer
    with open(c_path, 'r') as f:
        c = f.read()
    tmp_asm = get_tmp_file(c, extension='.s')
    stdout, stderr = run_command(f'gcc -O0 -c -S -o {tmp_asm} {c_path}')
    with open(tmp_asm, 'r') as f:
        asm = f.readlines()
    header = ''
    for line in asm:
        if ':' in line:
            break
        if 'file' in line:
            continue
        header += line
    with open(tmp_asm, 'r') as f:
        asm = f.read()
    os.remove(tmp_asm)
    _, _, footer = asm.partition('.cfi_endproc')
    return header, footer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Program to prepare reference C code to be compiled by a model.')
    parser.add_argument('--c-path', help="C directory or file. If it's a file path, then one function per line."
                                         "Otherwise, one function per file.")
    parser.add_argument('--model-path', help='Directory with the model (checkpoint, tokenizer)')
    parser.add_argument('--beam', type=int, default=1, help='Beam size')
    parser.add_argument('--top-n', type=int, default=1, help='Print top N hypotheses')
    parser.add_argument('--add-header-footer', action='store_true', help='Add assembly header and footer')
    parser.add_argument('--replace-true-false', action='store_true', help='Replace true/false in C with 1/0')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint_best.pt', help='Checkpoint path relative to'
                                                                                          'the model path, or absolute.'
                                                                                          'Default: checkpoint_best.pt')
    parser.add_argument('--config-file', type=str, help='Path to JSON file (instead of command line arguments)')

    args = parser.parse_args()

    orig_args = deepcopy(args)

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        args = argparse.Namespace(**config)
        args.config_file = orig_args.config_file
    else:
        config = vars(args)

    assert config['top_n'] <= config['top_n']

    # Set up logging etc
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    name = f'infer-beam{args.beam}-topn{args.top_n}-{timestamp}-{sha[:4]}-{extra_id[:4]}'
    eval_path = os.path.join(args.model_path, name)
    os.makedirs(eval_path)
    with open(os.path.join(eval_path, 'infer-config.json'), 'w') as f:
        json.dump(config, f)
    init_logging(os.path.join(eval_path, name + '.log'))
    print(eval_path)
    logging.info(args)

    logging.info('Preparing files...')

    # cat + tokenize (and found C files)
    cat_path, c_files = cat_tokenize(c_path=args.c_path, model_path=args.model_path, eval_path=eval_path,
                                     replace_true_false=args.replace_true_false)

    # interactive with different options
    logging.info('Running the model...')
    results = run_model(data_path=cat_path, model_path=args.model_path, beam=args.beam,
                        top_n=args.top_n, checkpoint_path=args.checkpoint_path)
    if os.path.isdir(args.c_path):
        examples_path = os.path.join(eval_path, 'examples')
        os.makedirs(examples_path)
        for c_file, result_list in zip(c_files, results):
            # TODO: insert header/footer?
            function_name = c_file.split(os.path.sep)[-2]
            new_path = os.path.join(examples_path, function_name)
            os.makedirs(new_path, exist_ok=True)
            shutil.copyfile(c_file, os.path.join(new_path, 'ref.c'))
            for idx, result in enumerate(result_list):
                result = result.replace('<newline>', '\n')
                if args.add_header_footer:
                    header, footer = get_asm_header_and_footer(c_file)
                    result = header + result + footer
                with open(os.path.join(new_path, function_name + f'-{idx+1}.s'), 'w')as f:
                    f.write(result)
    else:
        with open(args.c_path, 'r') as r:
            for orig_line, (idx, result) in zip(r.readlines(), enumerate(results[0])):

                    with open(os.path.join(eval_path, os.path.basename(args.c_path).split('.')[0] + f'-{idx+1}.s'), 'a+') as f:
                        if args.add_header_footer:
                            tmp = get_tmp_file(os.path.join(eval_path, args.c_path), extension='c')
                            header, footer = get_asm_header_and_footer(tmp)
                            os.remove(tmp)
                            result = header + result + footer
                        f.write(result + '\n')
