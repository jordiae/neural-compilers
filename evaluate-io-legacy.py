import os
from dataclasses import dataclass, asdict
from neural_compilers.utils.tokenization import GCC
from typing import *
import logging
import time
import uuid
import git
from bleu import list_bleu
from neural_compilers.utils.tokenization import PygmentsTokenizer
import lizard
import re, itertools
from copy import deepcopy
import sys
from io import StringIO
import contextlib
import json


def print(*args):
    logging.info(' '.join([str(arg) for arg in args]))


code_tokenizer = PygmentsTokenizer()


def eval_bleu(ref: str, hyp: str) -> float:
    return list_bleu([ref], [hyp], detok=False)

JUST_FUNC = True

BAD_CASES = []

BAD_EXAMPLES = {}

@dataclass
class FuncParameter:
    type_name: str
    var_name: str


@dataclass
class Signature:
    return_type: str
    func_name: str
    parameters: List[FuncParameter]


@dataclass
class Example:
    inp: List[str]
    out: str

@dataclass
class ExampleList:
    signature: Signature
    examples: List[Example]

@dataclass
class Array:
    var_name: str
    size: int

@dataclass
class Props:
    output: List[str]
    arrays: List[Array]


# From IO-EMBEDDINGS repo
def parse_file(path: str) -> Tuple[Signature, Example]:
    with open(path, 'r') as f:
        lines = f.readlines()
    # added hack for avoiding comments, macros, empty lines. TODO: review, improve
    lines = [line for line in lines if not line.startswith('//') and not line.startswith('#') and len(line.split()) > 0]
    signature = lines[0]
    signature_split = signature.split()
    return_type = signature_split[0]
    func_name = signature_split[1].split('(')[0]
    parameters = signature[signature.find('(') + 1:signature.find(')')]
    parsed_parameters = []
    for parameter in parameters.split(','):
        pointer = False
        if parameter.count('**') > 1:
            raise RuntimeError(parameter)
        if '*' in parameter:
            parameter = parameter.replace('*', '')
            pointer = True
        parameter = ' '.join(parameter.split())
        param_type, param_name = parameter.split()
        if pointer:
            param_type += '*'
        parsed_parameters.append(FuncParameter(type_name=param_type, var_name=param_name))
    parsed_signature = Signature(return_type=return_type, func_name=func_name, parameters=parsed_parameters)
    parsed_example = None
    return parsed_signature, parsed_example


def get_single_scanf(parameter: FuncParameter, declare: bool = True) -> str:
    scanf = []
    if parameter.type_name in ['int', 'bool']:
        if declare:
            scanf.append(f'  int {parameter.var_name};')
        scanf.append(f'  scanf("%d", &{parameter.var_name});')
    elif parameter.type_name == 'float':
        if declare:
            scanf.append(f'  float {parameter.var_name};')
        scanf.append(f'  scanf("%f", &{parameter.var_name});')
    elif parameter.type_name == 'char':
        if declare:
            scanf.append(f'  char {parameter.var_name};')
        scanf.append(f'  scanf("%c", &{parameter.var_name});')
    else:
        raise NotImplementedError(parameter.type_name)
    return '\n'.join(scanf)

def infer_size_from_code_or_examples(func_code: str, parameter: FuncParameter, examples) -> str:
    for line in func_code.splitlines():
        if 'for' in line:
            if parameter.var_name in line:
                before, _, after = line.partition('<')
                return after.split()[0].replace(';', '')
    raise RuntimeError('Cannot infer size from code')


# reverse_scalars: in simpl, scalars seem to be in reverse order
def get_scanf(signature: Signature, props: Props, func_code: str, reverse_scalars: bool = True) -> str:
    # hack to have n before arrays of size n
    scalar_scanfs = []
    array_scanfs = []
    for parameter in signature.parameters:

        if parameter.type_name.count('*') > 1:
            raise NotImplementedError(parameter.type_name)
        elif parameter.type_name.count('*') == 0:
            scalar_scanfs.append(get_single_scanf(parameter))
        else:  # == 1
            size = None
            for array in props.arrays:
                if array.var_name == parameter.var_name:
                    size = array.size
                    break
            scalar_type = parameter.type_name.replace('*', '')
            element = FuncParameter(type_name=scalar_type, var_name=parameter.var_name+'[idx]')
            single_scanf = get_single_scanf(element, declare=False)
            array_scanfs.append(f'  {parameter.type_name} {parameter.var_name};')
            array_scanfs.append(f"  {parameter.var_name} = ({parameter.type_name}) malloc({size}*sizeof({parameter.type_name.replace('*','')}));")
            array_scanfs.append('  int idx;')
            array_scanfs.append(f'  for (idx = 0; idx < {size}; idx++) ' + '{')
            array_scanfs.append('  ' + single_scanf)
            array_scanfs.append('  }')

    if len(scalar_scanfs) > 1 and reverse_scalars:
        scalar_scanfs.reverse()

    scanf = scalar_scanfs + array_scanfs

    return '\n'.join(scanf) + '\n'


def get_function_call(signature: Signature) -> str:
    res = ''
    if signature.return_type != 'void':
        res = f'  {signature.return_type} res;\n'
        res += '  res = '
    res += signature.func_name + '(' + ' '.join([par.var_name + ',' for par in signature.parameters])
    if len(signature.parameters) > 0:
        res = res[:-1]
    res += ');\n'
    return res


def get_single_printf(type_: str, var_name: str, trailing_space: bool = False) -> str:
    space = ' ' if trailing_space else ''
    if type_ in ['int', 'bool']:  # TODO: check bool
        return f'  printf("{space}%d", {var_name});'
    elif type_ == 'float':
        return f'  printf("{space}%f", {var_name});'
    elif type_ == 'char':
        return f'  printf({space}"%c", {var_name});'
    else:
        raise NotImplementedError(type_)


def print_newline() -> str:
    return '  printf("\\n");'


def get_printf(signature: Signature, props: Props) -> str:
    props = deepcopy(props)
    if signature.return_type == 'void':
        done_output = set([])
        printf = []
        for i in range(len(props.output)):
            result_type = None
            var_name = None
            for parameter in signature.parameters:
                # if parameter.var_name == props.output:
                if parameter.var_name in props.output and parameter.var_name not in done_output:
                    result_type = parameter.type_name.replace('*', '')
                    var_name = parameter.var_name
                    break
            size = None
            for array in props.arrays:
                # if array.var_name == props.output:
                if array.var_name in props.output and array.var_name not in done_output:
                    size = array.size
                    assert var_name == array.var_name
                    break
            if size is None:
                pass
            done_output.add(var_name)

            printf.append('  int idx;')
            printf.append('  int first = 0;')
            printf.append(f'  for (idx = 0; idx < {size}; idx++) ' + '{')
            printf.append('    if (first) {')
            printf.append('      ' + get_single_printf(result_type, var_name=var_name+'[idx]'))
            printf.append('        first = 0;')
            printf.append('    }')
            printf.append('    else {')
            printf.append('     ' + get_single_printf(result_type, var_name=var_name+'[idx]', trailing_space=True))
            printf.append('    }')
            printf.append('  }')
        printf = '\n'.join(printf) + '\n' + print_newline() + '\n'

    else:
        printf = get_single_printf(signature.return_type, var_name='res') + '\n' + print_newline() + '\n'
    return printf

def parse_props(props_str: str) -> Props:#, signature: Signature):
    # signature could be parsed from parse but we already have it
    if props_str.startswith('void') or props_str.count('output') > 0:
        result = []
        for l in props_str.splitlines():
            if l.startswith('output'):
                result.append(l.split()[1])
        if len(result) == 0:
            print('WARNING: Props output not found, using the only array instead')
            for l in props_str.splitlines()[1:]:
                _, var_name, size = l.split()
                var_name = var_name[:-1]
                result = [var_name]
                break
    else:
        result = ['res']
    arrays = []
    for l in props_str.splitlines()[1:]:
        if l.startswith('output'):
            continue
        _, var_name, size = l.split()
        var_name = var_name[:-1]
        array = Array(var_name=var_name, size=size)
        arrays.append(array)

    props = Props(output=result, arrays=arrays)
    return props


def contains_array(code: str) -> bool:
    for s in ['int*', 'char*', 'float*', 'int *', 'char *', 'float *']:
        if code.count(s) > 0:
            return True
    return False

def signature2standalone(signature: Signature, function_code: str, props: str, examples) -> Tuple[str,str]:
    ##### c_imp
    # props is only used if the return type is void, then we need to know which is the "result" of the function (side-effect)
    malloc_lib = '#include <stdlib.h>' if contains_array(function_code) else ''
    c_imp = f"#include <stdio.h>\n{malloc_lib}\n" + function_code + '\n'
    c_imp += '#include <math.h>\n#include <stdbool.h>\n'
    c_imp += 'int main() {\n'
    parsed_props = parse_props(props)
    scanf = get_scanf(signature, parsed_props, func_code=function_code)
    # print(scanf)
    c_imp += scanf
    function_call = get_function_call(signature)
    c_imp += '  ' + function_call
    printf = get_printf(signature, props=parsed_props)
    c_imp += printf

    c_imp += '\n  return 0;\n}\n'

    c_imp = c_imp.replace('None', 'n')

    # Force single declaration of idx
    before_first_idx, first_idx, after_first_idx = c_imp.partition('int idx;')
    c_imp = before_first_idx + first_idx + after_first_idx.replace('int idx;', '')
    #print(c_imp)

    def get_function_signature_string(f):
        s = ''
        for line in f.splitlines():
            if line.startswith('#'):
                continue
            if len(line.split()) == 0:
                continue
            else:
                s = line.strip().replace('{', '')
                break
        return s




    #### main_code (external)
    # props is only used if the return type is void, then we need to know which is the "result" of the function (side-effect)
    malloc_lib = '#include <stdlib.h>' if contains_array(function_code) else ''
    main_code = f"#include <stdio.h>\n{malloc_lib}\n"
    main_code += '#include <math.h>\n#include <stdbool.h>\n'
    main_code += f'extern {get_function_signature_string(function_code)};\n'
    main_code += 'int main() {\n'
    props = parse_props(props)
    scanf = get_scanf(signature, props, func_code=function_code)

    main_code += scanf
    function_call = get_function_call(signature)
    main_code += '  ' + function_call
    printf = get_printf(signature, props=props)
    main_code += printf

    main_code += '\n  return 0;\n}\n'

    # Force single declaration of idx
    before_first_idx, first_idx, after_first_idx = main_code.partition('int idx;')
    main_code = before_first_idx + first_idx + after_first_idx.replace('int idx;', '')

    main_code = main_code.replace('None', 'n')

    return c_imp, main_code


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def run_python_script(name: str, path: str) -> str:
    previous_dir = os.getcwd()
    os.chdir(path)

    with stdoutIO() as s:
        try:
            exec(open(name).read())
        except BaseException as e:
            pass
    os.chdir(previous_dir)

# 1 - min_so_far_subtracted etc have errors in the L2 files.
# 2- integers must come first to guarantee that array sizes are initialized before arrays. however, in the examples these is not respected
# so we reorder
# so, scalars (in order) then arrays (in order)
# assume only input is affected
def get_examples(example_path: str, use_simpl_instead_of_L2: bool, scalars_first: bool, signature: Signature) -> List[Tuple[str, str]]:


    if use_simpl_instead_of_L2:
        with open(os.path.join(example_path, 'simpl'), 'r') as f:
            simpl = f.read()
            data = simpl2json(simpl, signature=signature)

            simpl_header = [l for l in simpl.split('\n') if "fun" in l][0]
            diff_num_parameters = len(simpl_header.replace('fun', '').replace('->','').split()) - len(signature.parameters)
            if diff_num_parameters != 0:
                # simpl includes length of output
                if diff_num_parameters == 1:
                    import re
                    for i in range(len((data['contents']['examples']))):

                        data['contents']['examples'][i] = re.sub(' -?\d+\)',')', data['contents']['examples'][i])
                else:

                    # simpl includes length of output
                    if signature.return_type == 'void':
                        import re
                        for i in range(len((data['contents']['examples']))):
                            data['contents']['examples'][i] = re.sub(' -?\d+\)', ')', data['contents']['examples'][i])
                    # simpl includes  length for each array, even if according to the c implementation they are equal
                    import re
                    for i in range(len((data['contents']['examples']))):
                        # remove extra Ns

                        c = itertools.count()
                        data['contents']['examples'][i] = re.sub('\] -?\d+ \[', lambda x: x.group() if not next(c) else '] [', data['contents']['examples'][i])
                        data['contents']['examples'][i] = re.sub(r"(\]) (-?\d+) (-?\d+) (-?\d+) (\[)", r"\1 \3 \5", data['contents']['examples'][i])



    else:
        with open(os.path.join(example_path, 'L2'), 'r') as f:
            data = json.load(f)

    parsed_examples = []
    for example in data['contents']['examples']:
        # "(f 1) -> 1", "(f 4) -> 36", "(f 4) -> 36", "(f 1) -> 1"
        inp, _, out = example.partition('->')
        inp = ' '.join(inp.strip()[2:-1].split())
        out = ' '.join(out.split())

        def parse(text):
            parsed = ''
            i = 0
            in_array = False
            while i < len(text):
                current_char = text[i]
                if len(current_char.split()) == 0:
                    if in_array:
                        parsed += ' '
                    else:
                        parsed += '\n'
                elif current_char == '[':
                    parsed += '\n'
                    in_array = True
                elif current_char == ']':
                    parsed += '\n'
                    in_array = False
                else:
                    parsed += current_char
                i += 1
            return parsed

        def parse_scalars_first(text):
            parsed_scalars = ''
            parsed_arrays = ''
            i = 0
            in_array = False
            while i < len(text):
                current_char = text[i]
                if len(current_char.split()) == 0:
                    if in_array:
                        parsed_arrays += ' '
                    else:
                        parsed_scalars += '\n'
                elif current_char == '[':
                    parsed_arrays += '\n'
                    in_array = True
                elif current_char == ']':
                    parsed_arrays += '\n'
                    in_array = False
                else:
                    if in_array:
                        parsed_arrays += current_char
                    else:
                        parsed_scalars += current_char
                i += 1
            return parsed_scalars + parsed_arrays

        if scalars_first:
            parsed_inp = parse_scalars_first(inp)
        else:
            parsed_inp = parse(inp)
        parsed_out = parse(out)
        parsed_examples.append((parsed_inp, parsed_out))
    return parsed_examples




def get_asm_header_footer_body(asm_path: str) -> Tuple[str, str, str]:
    with open(asm_path, 'r') as f:
        asm = f.readlines()
    header = ''
    for line in asm:
        if ':' in line:
            break
        header += line
    with open(asm_path, 'r') as f:
        asm = f.read()
    body, _, footer = asm.partition('.cfi_endproc')
    body = body[len(header):]
    return header, footer, body


def simpl2json(simpl: str, signature: Signature) -> Dict:
    '''

    Examples
Examples
{6,0,4,8,7,6,4,7,5,9,3,8,2,4},14,{2,1,9,4,8,9,2,4,1,1,10,5,7,8},14,{0,0,0,0,0,0,0,0,0,0,0,0,0,0},14 -> {-1,0,-8,-3,-7,-8,-1,-3,0,0,-9,-4,-6,-7};
{5,6,5,9},4,{10,3,8,7},4,{0,0,0,0},4 -> {-9,-2,-7,-6};
{8,4,0,8,0,1,6,10,10,0,9,7,5,3,5,1},16,{3,9,3,3,2,8,7,1,1,5,8,7,1,4,8,4},16,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},16 -> {-2,-8,-2,-2,-1,-7,-6,0,0,-4,-7,-6,0,-3,-7,-3};
{8,5,8,3},4,{9,8,9,4},4,{0,0,0,0},4 -> {-8,-7,-8,-3};
{1,9,6,5,9,3,4,2,3,2,0,9,10,4,7,1},16,{1,10,2,2,0,1,8,10,6,8,4,8,3,3,10,9},16,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},16 -> {0,-9,-1,-1,0,0,-7,-9,-5,-7,-3,-7,-2,-2,-9,-8};
{9,4,7,7,10,10,5,1,5,9,1,7,9,10},14,{5,3,3,0,4,1,3,5,2,5,6,0,1,2},14,{0,0,0,0,0,0,0,0,0,0,0,0,0,0},14 -> {-4,-2,-2,0,-3,0,-2,-4,-1,-4,-5,0,0,-1};
{0,9,10,8,9,10,1,0},8,{1,10,3,9,9,1,6,1},8,{0,0,0,0,0,0,0,0},8 -> {0,-9,-2,-8,-8,0,-5,0};
{1,0,9,0,3,2,1,7,3,0,10,0},12,{8,6,9,1,4,1,3,1,10,4,5,6},12,{0,0,0,0,0,0,0,0,0,0,0,0},12 -> {-7,-5,-8,0,-3,0,-2,0,-9,-3,-4,-5};
{0,8,7,0,9,1},6,{6,3,4,5,7,9},6,{0,0,0,0,0,0},6 -> {-5,-2,-3,-4,-6,-8};

["(f [] []) -> []", "(f [6 0 4 8 7 6 4 7 5 9 3 8 2 4] [2 1 9 4 8 9 2 4 1 1 10 5 7 8]) -> [-1 0 -8 -3 -7 -8 -1 -3 0 0 -9 -4 -6 -7]",

    '''
    L2_examples = []
    lines = simpl.splitlines()[1:]
    simpl_header = [l for l in simpl.split('\n') if "fun" in l][0]
    for line in lines:
        if len(line.split()) == 0:
            break
        L2 = line.strip().replace('{', '[').replace('}', ']').replace(',', ' ').replace(';', '')
        L2 = '(f ' + L2.replace(' ->', ') ->')
        L2_examples.append(L2)
        # hack to have n before arrays of size n!
        diff_num_parameters = len(simpl_header.replace('fun', '').replace('->','').split()) - len(signature.parameters)
        if diff_num_parameters != 0:
            # simpl includes length of output
            pass
    return {'contents': {'examples': L2_examples}}



def run_io(c_code: str, example_path: str, just_func_code: str, main_code: str, signature: Signature, predictions_path: str,  use_simpl_instead_of_L2: bool) -> Tuple[bool, bool, bool, float]:
    try:
        examples = get_examples(example_path, use_simpl_instead_of_L2=use_simpl_instead_of_L2, scalars_first=True, signature=signature)
    except FileNotFoundError as e:
        return False, False, False, 0.0  # benchmark ok, model ok, syntax model ok, BLEU model

    func_name = example_path.split(os.sep)[-1]
    dir_ = os.path.join(predictions_path, func_name)

    # Run with gcc:sc
    from neural_compilers.utils.utilities import get_tmp_file, run_command
    tmp_c = get_tmp_file(c_code, extension='.c', dir=dir_)
    output = tmp_c[:-2] + '.x'
    stdout, stderr = run_command(f'gcc -O0 -x c -o {output} {tmp_c}')
    print(stderr)
    gcc_corrects = 0
    bad_examples_in_benchmark = 0
    for idx, (in_, out_) in enumerate(examples):
        if os.path.basename(example_path) in BAD_EXAMPLES and idx in BAD_EXAMPLES[os.path.basename(example_path)]:
            bad_examples_in_benchmark += 1
            continue
        # stdout, stderr = run_command(f'./{output} {tmp_c}', stdin=
        prefix_ex = './' if not os.path.isabs(output) else ''
        try:
            stdout, stderr = run_command(f'{prefix_ex}{output}', stdin=in_)
        except FileNotFoundError as e:
            return False, False, False, 0.0
        if stdout.strip() == out_.strip():
            gcc_corrects += 1

    print(example_path, 'GCC: ', f'{gcc_corrects}/{len(examples) - bad_examples_in_benchmark}')
    print()
    os.remove(tmp_c)
    os.remove(output)

    # first compile with gcc to get header and footer
    tmp_c = get_tmp_file(just_func_code, extension='.c', dir=dir_)
    output = tmp_c[:-2] + '.s'
    stdout, stderr = run_command(f'gcc -O0 -c -S -o {output} {tmp_c}')
    asm_header, asm_footer, asm_body = get_asm_header_footer_body(output)
    func_name = example_path.split(os.sep)[-1]
    import glob
    max_model_corrects = 0
    best_bleu = 0.0
    best_syntax = False
    for idx, hypothesis in reversed(list(enumerate(sorted(glob.glob(os.path.join(predictions_path, func_name, f'{func_name}*.s')))))):
        print('Hypothesis:', idx+1)
        print(hypothesis)
        hypothesis_ = open(hypothesis).read()
        model_assembly = asm_header + hypothesis_ + asm_footer
        tmp_s = get_tmp_file(model_assembly, extension='.s', dir=dir_)
        output = tmp_c[:-2] + '.x'
        main_ = get_tmp_file(main_code, extension='.c', dir=dir_)
        stdout, stderr = run_command(f'gcc -O0 -o {output} {main_} {tmp_s}')


        model_corrects = 0
        bad_examples_in_benchmark = 0
        not_compiled = False
        for idx, (in_, out_) in enumerate(examples):
            if os.path.basename(example_path) in BAD_EXAMPLES and idx in BAD_EXAMPLES[os.path.basename(example_path)]:
                bad_examples_in_benchmark += 1
                continue
            try:
                prefix_ex = './' if not os.path.isabs(output) else ''
                stdout, stderr = run_command(f'{prefix_ex}{output}', stdin=in_, timeout=5)
                if stdout.strip() == out_:
                    model_corrects += 1
            except BaseException as e:
                if isinstance(e, TimeoutError):
                    break
                not_compiled = True
                break
        ref_tok = ' '.join(code_tokenizer.tokenize(programs=asm_body, lang='asm'))
        hyp_tok = hypothesis_.replace('\n', '<newline>')
        bleu_score = eval_bleu(ref=ref_tok, hyp=hyp_tok)
        print('BLEU =', bleu_score)
        if bleu_score > best_bleu:
            best_bleu = bleu_score
        print('SYNTAX', 'INCORRECT' if not_compiled else 'CORRECT')
        if not not_compiled:
            if not best_syntax:
                best_syntax = True
            print(example_path, 'IO: ', f'{model_corrects}/{len(examples) - bad_examples_in_benchmark}')
        else:
            print(example_path, 'IO: N/A (Error:', stderr, ')')
        print()
        if not not_compiled:
            os.remove(tmp_s)
            os.remove(output)

        if model_corrects > max_model_corrects:
            max_model_corrects = model_corrects
    return gcc_corrects == len(examples), max_model_corrects == gcc_corrects and gcc_corrects > 0, best_syntax, best_bleu


def run(synthesis_eval_path: str, predictions_path: str):
    standaloner_code_works = 0
    total = 0
    none_in_code = 0
    benchmark_oks = 0
    model_oks = 0
    syntax_oks = 0
    bleu = 0.0
    for idx, example in enumerate(sorted(os.listdir(os.path.join(synthesis_eval_path, 'examples')))):
        example_path = os.path.join(synthesis_eval_path, 'examples', example)
        if example.startswith('__') or not os.path.isdir(example_path) or example in BAD_CASES:
            continue
        total += 1
        c_path = os.path.join(example_path, 'ref.c')
        parsed_signature, _ = parse_file(c_path)
        with open(c_path, 'r') as c:
            c_code = c.read()
        props_path = os.path.join(example_path, 'props')
        with open(props_path, 'r') as p:
            props = p.read()
        try:
            c_imp, main_code = signature2standalone(parsed_signature, c_code, props, examples=get_examples(example_path, use_simpl_instead_of_L2=True, scalars_first=True, signature=parsed_signature))
            if c_imp.count('None') > 0:
                none_in_code += 1
            gcc = GCC(print_stderr=False)
            if len(gcc.compile(c_imp).splitlines()) > 1:
                standaloner_code_works += 1
            print('-------------------')

            benchmark_ok, model_ok, best_syntax, best_bleu = run_io(c_imp, example_path, just_func_code=c_code, main_code=main_code, signature=parsed_signature, predictions_path=predictions_path, use_simpl_instead_of_L2=True)
            if not benchmark_ok:
                benchmark_ok, model_ok, best_syntax, best_bleu = run_io(c_imp, example_path, just_func_code=c_code, main_code=main_code, signature=parsed_signature,
                       predictions_path=predictions_path, use_simpl_instead_of_L2=False)
            if benchmark_ok:
                benchmark_oks += benchmark_ok
                model_oks += model_ok
                syntax_oks += best_syntax
                bleu += best_bleu
                def str_ok(b):
                    return "OK" if b else "NOT OK"
                print('Benchmark OK!')
                complexity_ref = lizard.analyze_file(c_path).__dict__['function_list'][0].__dict__
                cyclomatic = complexity_ref['cyclomatic_complexity']
                nloc = complexity_ref['nloc']
                tokens = complexity_ref['token_count']
                params = len(complexity_ref['parameters'])
                pointers = complexity_ref['long_name'].count('*')
                print(f'{example}: IO = {str_ok(model_ok)} | SYNTAX = {str_ok(best_syntax)} | BLEU = {best_bleu}'
                      f' | C_NLOC = {nloc} | C_TOKENS = {tokens} | C_CYCLO = {cyclomatic} | PARAMS = {params} | POINTERS = {pointers}')
            else:
                print('Benchmark NOT OK!')
        except (NotImplementedError, FileNotFoundError) as e:
            print('Benchmark NOT OK!')

    print('\nBenchmark ok:', benchmark_oks, 'of', total)
    print('IO ok:', model_oks, 'of', benchmark_oks)
    print('Syntax ok:', syntax_oks, 'of', benchmark_oks)
    print('Avg BLEU:', bleu/benchmark_oks)


if __name__ == '__main__':
    import argparse
    from neural_compilers.utils.utilities import init_logging
    import os
    parser = argparse.ArgumentParser('IO Evaluator')
    parser.add_argument('--synthesis-eval-path', type=str)
    parser.add_argument('--predictions-path', type=str)
    args = parser.parse_args()
    # Set up logging etc
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    name = f'eval-io-legacy-{timestamp}-{sha[:4]}-{extra_id[:4]}'
    eval_path = os.path.join(os.path.dirname(args.predictions_path), name)
    os.mkdir(eval_path)
    init_logging(os.path.join(eval_path, name + '.log'))
    print(args)
    run(synthesis_eval_path=args.synthesis_eval_path, predictions_path=args.predictions_path)
    print(os.path.join(eval_path, name + '.log'))
