import argparse
import os
from neural_compilers.utils.utilities import run_command, get_tmp_path
from typing import List
from neural_compilers.utils.constants import NEWLINE_ESCAPE
from typing import Dict
import git
import time
import uuid
import json


def save_eval(system_output_path: str, eval_script: str, config: Dict, results: Dict):
    # Set up logging etc
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    name = f'{eval_script}-{timestamp}-{sha[:4]}-{extra_id[:4]}'
    if os.path.isfile(system_output_path):
        system_output_path = system_output_path + '_' + name
    eval_path = os.path.join(system_output_path, name)
    os.makedirs(eval_path)
    with open(os.path.join(eval_path, 'eval-config.json'), 'w') as f:
        json.dump(config, f)
    with open(os.path.join(eval_path, 'results.json'), 'w') as f:
        json.dump(results, f)


class Evaluator:
    def asm_is_valid(self, asm: str) -> bool:
        raise NotImplementedError


class GASSyntaxEvaluator:
    def asm_is_valid(self, asm: str) -> bool:
        asm = asm.replace(NEWLINE_ESCAPE, '\n')
        # make gcc read from stdin and write to stdout
        # print(asm)
        # Note: We CANNOT print the assembler output to stdout. See:
        # https://stackoverflow.com/questions/47181017/why-cant-i-pipe-assembler-output-to-stdout
        # stdout, stderr = run_command(f'gcc -c -x assembler -o /dev/stdout -', stdin=asm)
        tmp_path = get_tmp_path()
        stdout, stderr = run_command(f'gcc -c -x assembler -o {tmp_path} -', stdin=asm)
        if os.path.exists(tmp_path):  # compiled
            os.remove(tmp_path)
            return True
        # Print wrong assembly (for error analysis)
        print('-------------')
        print('INVALID ASSEMBLY!')
        print('ERROR:')
        print(stderr)
        print('ASSEMBLY:')
        print()
        print(asm)
        print('-------------')
        print()
        return False


    def aggregate(self, corrects: List[bool]) -> Dict:
        return {'syntactic_accuracy': sum(corrects)/len(corrects), 'valid_compilations': sum(corrects),
                'total': len(corrects)}

