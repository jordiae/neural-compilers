from neural_compilers.eval.evaluator import GASSyntaxEvaluator
import argparse
from typing import Dict
from tokenizers import Tokenizer
import os
from typing import Optional
from pathlib import Path
from copy import deepcopy
import json
from neural_compilers.eval.evaluator import save_eval

SUBWORD_TAG = '##'


def evaluate(system_output_path: str, decode_subwords: bool, system_path: Optional[str]=None) -> Dict:
    evaluator = GASSyntaxEvaluator()
    results = []

    if os.path.isfile(system_output_path):

        with open(system_output_path, 'r') as f:
            for compilation in f.readlines():
                if decode_subwords:
                    compilation = compilation.replace(f' {SUBWORD_TAG}', '')
                results.append(evaluator.asm_is_valid(compilation))
    else:
        for file in sorted(Path(system_output_path).rglob('*.c')):
            with open(file, 'r') as f:
                compilation = f.read()
                if decode_subwords:
                    compilation = compilation.replace(f' {SUBWORD_TAG}', '')
                results.append(evaluator.asm_is_valid(compilation))

    return evaluator.aggregate(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generation')
    parser.add_argument('--system-output-path', type=str, help="Path to compilations. C directory or file."
                                                               "If it's a file path, then one function per line."
                                                                "Otherwise, one function per file.")
    parser.add_argument('--system-path', type=str, default=None, help='Path to system (tokenizer)')
    parser.add_argument('--no-decode-subwords', action='store_true')
    parser.add_argument('--config-file', type=str, help='Path to JSON file (instead of command line arguments)')

    args = parser.parse_args()

    orig_args = deepcopy(args)

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        args = argparse.Namespace(**config)
    else:
        config = vars(args)
    args.decode_subwords = not args.no_decode_subwords

    results = evaluate(args.system_output_path, args.decode_subwords, args.system_path)
    print(results)

    save_eval(system_output_path=config['system_output_path'], eval_script=__file__, config=config,
              results=dict(syntax=results))
