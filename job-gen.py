import os
import uuid
import time
from typing import Optional


def get_template(name: str, system: str, config_path: str, job_type: str, n_gpus: Optional[int] = None) -> str:
    if system == 'cluster' and job_type in ['train', 'infer']:
        template = f'''#!/bin/bash
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --mem=20G
#SBATCH --output=slurm_logs/{name}_%j.log'''
    elif system == 'calcula':
        template = f'''#!/bin/bash
#SBATCH -p veu # Partition to submit to
#SBATCH --mem=20G
#SBATCH --output=slurm_logs/{name}_%j.log
source ~/.bashrc
'''
    elif system in ['local', 'workstation']:
        template = ''
    else:
        raise NotImplementedError()
    if n_gpus == 0:
        template = template.replace('SBATCH --gres=gpu:0\n ', '')
    template += '\nsource venv/bin/activate'
    if job_type == 'train':
        cmd = f'bash {config_path}'
    elif job_type == 'data':
        cmd = f'python prepare-train-data.py --config-file {config_path}'
    elif job_type == 'infer':
        cmd = f'python infer.py --config-file {config_path}'
    elif job_type == 'eval-io':
        cmd = f'python evaluate-io.py --config-file {config_path}'
    elif job_type == 'eval-bleu':
        cmd = f'python evaluate-bleu.py --config-file {config_path}'
    elif job_type == 'eval-syntax':
        cmd = f'python evaluate-syntax.py --config-file {config_path}'
    else:
        raise NotImplementedError()
    template += f'\n{cmd}\n'
    return template


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Job generator')
    parser.add_argument('--system', type=str, choices=['cluster', 'local', 'workstation'],
                        help='Where the run will be run')
    parser.add_argument('--config-path', type=str, help='Path to config')
    parser.add_argument('--job-type', type=str, choices=['data', 'train', 'infer', 'eval-io', 'eval-bleu',
                                                         'eval-syntax'], help='Whether it is a data, training,'
                                                                              'inference, or evaluation job')
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    extra_id = uuid.uuid4().hex
    stamp = f'{timestamp}-{extra_id[:4]}'
    job_name = f'{args.system}-{os.path.basename(args.config_path)}-{stamp}'
    job_path = os.path.join('jobs', job_name + '.sh')
    template = get_template(job_name, args.system, args.config_path, args.job_type, args.gpus)

    with open(job_path, 'w') as f:
        f.write(template)
    if args.system == 'local':
        cmd = 'bash'
    else:
        cmd = 'sbatch'
    cmd += ' ' + job_path
    print(cmd)
