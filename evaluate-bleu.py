from bleu import list_bleu
import os
from neural_compilers.utils.tokenization import PygmentsTokenizer
from pathlib import Path
import json
from copy import deepcopy
from neural_compilers.eval.evaluator import save_eval

code_tokenizer = PygmentsTokenizer()


def eval_bleu(ref: str, hyp: str) -> float:
    return list_bleu(ref, hyp, detok=False)


def eval_line_by_line_file(ref_path: str, out_path: str):
    score = 0.0
    count = 0
    with open(ref_path, 'r') as refs, open(out_path, 'r') as hyps:
        for ref, hyp in zip(refs.readlines(), hyps.readlines()):
            count += 1
            ref_tok = ' '.join(code_tokenizer.tokenize(programs=ref, lang='asm'))
            hyp_tok = ' '.join(code_tokenizer.tokenize(programs=hyp, lang='asm'))
            score += eval_bleu(ref_tok, hyp_tok)
    return score/count


def eval_dir(ref_path: str, out_path: str):
    score = 0.0
    count = 0
    asm_files_ref = sorted(Path(ref_path).rglob('*.s'))
    asm_files_out = sorted(Path(out_path).rglob('*.s'))
    for asm_file_ref, asm_file_out in zip(asm_files_ref, asm_files_out):
        with open(asm_file_ref, 'r') as ref, open(asm_file_out, 'r') as hyp:
            ref = ref.read()
            hyp = hyp.read()
            count += 1
            ref_tok = ' '.join(code_tokenizer.tokenize(programs=ref, lang='asm'))
            hyp_tok = ' '.join(code_tokenizer.tokenize(programs=hyp, lang='asm'))
            score += eval_bleu(ref_tok, hyp_tok)
    return score/count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('BLEU evaluator')
    parser.add_argument('--ref-path', help="Reference C directory or file. If it's a file path, then one function per"
                                           "line. Otherwise, one function per file.")
    parser.add_argument('--out-path', help="System output ASM directory or file. If it's a file path, then one function"
                                           "per line. Otherwise, one function per file.")
    parser.add_argument('--config-file', type=str, help='Path to JSON file (instead of command line arguments)')

    args = parser.parse_args()

    orig_args = deepcopy(args)

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        args = argparse.Namespace(**config)
    else:
        config = vars(args)

    if os.path.isdir(args.ref_path):
        res = eval_dir(config['ref_path'], config['out_path'])
    else:
        res = eval_line_by_line_file(config['ref_path'], config['out_path'])
    save_eval(system_output_path=config['out_path'], eval_script=__file__, config=config, results=dict(bleu=res))
    print(res)
