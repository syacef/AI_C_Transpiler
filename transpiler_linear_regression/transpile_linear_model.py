import joblib
import numpy as np
import subprocess
import sys
import os
import math

MODEL_PATH = 'model.joblib'
C_FILE = 'model.c'
EXE = 'model'

def float_literal(x):
    v = np.float32(x)

    s = f"{v:.8g}"

    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'

    return s + 'f'

def generate_c_code(coefs, intercept, sample_features):
    n = len(coefs)
    lines = []
    lines.append('#include <stdio.h>')
    lines.append('')
    lines.append('float prediction(float *features, int n_feature) {')
    lines.append(f'    float result = {float_literal(intercept)};')
    for i, c in enumerate(coefs):
        lines.append(f'    result += features[{i}] * {float_literal(c)};')
    lines.append('    return result;')
    lines.append('}')
    lines.append('')
    features_init = ', '.join(f'{float_literal(x)}' for x in sample_features)
    lines.append('int main() {')
    lines.append(f'    float features[] = {{ {features_init} }};')
    lines.append('    int n_feature = sizeof(features)/sizeof(features[0]);')
    lines.append('    float pred = prediction(features, n_feature);')
    lines.append('    printf("%f\\n", pred);')
    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)

def compile_c(c_file, out_exe):
    cmd = ['gcc', '-O3', '-std=c99', '-pedantic', '-Werror', '-Wall', '-Werror','-Wvla', c_file, '-o', out_exe]
    print('Compile command:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        print('Compilation succeeded ->', out_exe)
        return True
    except Exception as e:
        print('Compilation failed:', e)
        return False

def run_exe(out_exe):
    try:
        out = subprocess.check_output(['./' + out_exe], universal_newlines=True)
        return float(out.strip())
    except Exception as e:
        print('Running executable failed:', e)
        return None

def almost_equal(a, b, tol=1e-3):
    return abs(a - b) <= tol or math.isclose(a, b, rel_tol=1e-6)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Please train and save the model first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    coefs = np.asarray(model.coef_).tolist()
    intercept = float(model.intercept_)

    sample_features = None
    if os.path.exists('houses.csv'):
        import pandas as pd
        df = pd.read_csv('houses.csv')
        sample_features = df[['size', 'nb_rooms']].iloc[0].values.tolist()
    else:
        sample_features = [2100.0, 3.0]

    print('Using sample features:', sample_features)

    c_code = generate_c_code(coefs, intercept, sample_features)
    with open(C_FILE, 'w') as f:
        f.write(c_code)
    print(f'Wrote C source to {C_FILE}')

    compiled = compile_c(C_FILE, EXE)
    if not compiled:
        print('Compilation failed; you can try the printed command manually.')
        sys.exit(1)

    pred_c = run_exe(EXE)
    if pred_c is None:
        print('Could not get prediction from compiled binary.')
        sys.exit(1)

if __name__ == '__main__':
    main()
