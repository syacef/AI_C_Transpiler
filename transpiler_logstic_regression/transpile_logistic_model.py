import joblib
import numpy as np
import subprocess
import sys
import os
import pandas as pd


MODEL_PATH = 'model.joblib'
C_FILE = 'model_logistic.c'
EXE = 'model_logistic'

def float_literal(x):
    v = np.float32(x)
    s = f"{v:.8g}"
    if ('.' in s) or ('e' in s) or ('E' in s):
        return s + 'f'
    else:
        return s + '.0f'

def generate_c_code(coefs, intercept, sample_features, classes_map):
    lines = []
    lines.append('#include <stdio.h>')
    lines.append('#include <math.h>')
    lines.append('')
    lines.append('static float sigmoid(float x) {')
    lines.append('    return 1.0f / (1.0f + expf(-x));')
    lines.append('}')
    lines.append('')
    lines.append('float predict_prob(float *features, int n_feature) {')
    lines.append(f'    float z = {float_literal(intercept)};')
    for i, c in enumerate(coefs):
        lines.append(f'    z += features[{i}] * {float_literal(c)};')
    lines.append('    return sigmoid(z);')
    lines.append('}')
    lines.append('')
    lines.append('int predict_class(float *features, int n_feature) {')
    lines.append('    float p = predict_prob(features, n_feature);')
    lines.append('    return (p >= 0.5f) ? 1 : 0;')
    lines.append('}')
    lines.append('')
    features_init = ', '.join(float_literal(x) for x in sample_features)
    lines.append('int main() {')
    lines.append(f'    float features[] = {{ {features_init} }};')
    lines.append('    int n_feature = sizeof(features)/sizeof(features[0]);')
    lines.append('    float prob = predict_prob(features, n_feature);')
    lines.append('    int cls = predict_class(features, n_feature);')
    if classes_map is not None:
        label0 = classes_map[0]
        label1 = classes_map[1]
        try:
            int(label0)
            int(label1)
            lines.append(f'    int labels[2] = {{ {int(label0)}, {int(label1)} }};')
            lines.append('    printf("prob=%f class=%d\\n", prob, labels[cls]);')
        except Exception:
            lines.append('    printf("prob=%f class_index=%d\\n", prob, cls);')
    else:
        lines.append('    printf("prob=%f class_index=%d\\n", prob, cls);')
    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)

def compile_c(c_file, out_exe):
    cmd = ['gcc', '-O3', '-std=c99', '-pedantic', '-Werror', '-Wall', '-Werror','-Wvla', c_file, '-o', out_exe, '-lm']
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
        return out.strip()
    except Exception as e:
        print('Running executable failed:', e)
        return None

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Please train and save a LogisticRegression model to {MODEL_PATH} first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    if hasattr(model, 'coef_') is False:
        print('Loaded object has no coef_; is it a sklearn LogisticRegression?')
        sys.exit(1)

    coefs = np.asarray(model.coef_)
    if coefs.ndim == 2 and coefs.shape[0] == 1:
        coefs = coefs[0].tolist()
    else:
        print('Only binary logistic regression (one-vs-rest) is supported by this transpiler.')
        sys.exit(1)

    intercept = float(model.intercept_.ravel()[0])

    sample_features = None
    data_path = os.path.join('..', 'data', 'diabetes.csv')
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Outcome' in numeric:
                numeric.remove('Outcome')
            if len(numeric) >= 1:
                vals = df[numeric].iloc[0].values.tolist()
                if len(vals) > len(coefs):
                    sample_features = vals[:len(coefs)]
                elif len(vals) < len(coefs):
                    sample_features = vals + [0.0] * (len(coefs) - len(vals))
                else:
                    sample_features = vals
        except Exception:
            sample_features = None

    classes_map = None
    if hasattr(model, 'classes_'):
        classes = list(model.classes_)
        if len(classes) == 2:
            classes_map = classes
        else:
            print('Multi-class logistic regression not supported by this transpiler.')
            sys.exit(1)

    print('Using sample features:', sample_features)
    c_code = generate_c_code(coefs, intercept, sample_features, classes_map)
    with open(C_FILE, 'w') as f:
        f.write(c_code)
    print(f'Wrote C source to {C_FILE}')

    compiled = compile_c(C_FILE, EXE)
    if not compiled:
        print('Compilation failed; you can try the printed command manually.')
        sys.exit(1)

    out = run_exe(EXE)
    if out is None:
        print('Could not run compiled binary.')
        sys.exit(1)

    print('Output from binary:', out)


if __name__ == '__main__':
    main()
