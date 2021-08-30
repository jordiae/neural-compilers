import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import re
from scipy.stats import pearsonr

DATA = 'output/table.tex'

if __name__ == '__main__':
    with open(DATA, 'r') as f:
        content = f.read()
    content = content.lower().replace('\\cmark', '1').replace('\\xmark', '0')
    content = re.sub(r'(\\textsc{)(\w+)(})', '\\2', content)
    with open(DATA + '.tmp.tex', 'w') as f:
        f.write(content)
    df = Table.read(DATA + '.tmp.tex').to_pandas()
    print(df[df.columns[1:]].corr()['io'][:])

    import numpy as np

    rho = df[df.columns[1:]].corr()
    pval = df[df.columns[1:]].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [0.01, 0.05, 0.1] if x <= t]))
    p = pval.applymap(lambda x: ('*' if x <= 0.05 else '') + f' ({x:.2E})')
    print((rho.round(3).astype(str) + p)['io'])
