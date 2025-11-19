import joblib
import numpy as np
import os
import sys
import subprocess
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODEL_PATH = 'model_dt.joblib'
C_FILE = 'model_dt.c'
EXE = 'model_dt'

def float_literal(x):
    v = np.float32(x)
    s = f"{v:.8g}"
    if ('.' in s) or ('e' in s) or ('E' in s):
        return s + 'f'
    else:
        return s + '.0f'

def emit_tree_code(tree, feature_names=None, is_classifier=True):
    left = tree.children_left
    right = tree.children_right
    threshold = tree.threshold
    features_idx = tree.feature
    value = tree.value

    def node_to_code(node, indent=4):
        pad = ' ' * indent
        if left[node] == right[node]:
            if is_classifier:
                counts = value[node][0]
                maj_idx = int(np.argmax(counts))
                return pad + f'return {maj_idx};\n'
            else:
                val = float(value[node][0][0])
                return pad + f'return {float_literal(val)};\n'
        else:
            feat = features_idx[node]
            thr = threshold[node]
            s = ''
            s += pad + f'if (features[{feat}] <= {float_literal(thr)}) {{\n'
            s += node_to_code(left[node], indent + 4)
            s += pad + '} else {\n'
            s += node_to_code(right[node], indent + 4)
            s += pad + '}\n'
            return s

    return node_to_code(0, indent=4)

def generate_c(tree, sample_features, classes=None, is_classifier=True):
    lines = []
    lines.append('#include <stdio.h>')
    lines.append('')
    if is_classifier:
        lines.append('int predict(float *features, int n_feature) {')
    else:
        lines.append('float predict(float *features, int n_feature) {')
    lines.append(emit_tree_code(tree, is_classifier=is_classifier))
    lines.append('}')
    lines.append('')
    features_init = ', '.join(float_literal(x) for x in sample_features)
    lines.append('int main() {')
    lines.append(f'    float features[] = {{ {features_init} }};')
    lines.append('    int n_feature = sizeof(features)/sizeof(features[0]);')
    if is_classifier:
        lines.append('    int cls = predict(features, n_feature);')
        if classes is not None:
            try:
                labels = [int(c) for c in classes]
                lines.append(f'    int labels[] = {{ {labels[0]}, {labels[1]} }};')
                lines.append('    printf("%d\\n", labels[cls]);')
            except Exception:
                lines.append('    printf("%d\\n", cls);')
        else:
            lines.append('    printf("%d\\n", cls);')
    else:
        lines.append('    float y = predict(features, n_feature);')
        lines.append('    printf("%f\\n", y);')
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
        return out.strip()
    except Exception as e:
        print('Running executable failed:', e)
        return None

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Run train_decision_tree.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    is_classifier = isinstance(model, DecisionTreeClassifier)
    is_regressor = isinstance(model, DecisionTreeRegressor)
    if not (is_classifier or is_regressor):
        print('Only sklearn DecisionTreeClassifier or DecisionTreeRegressor are supported.')
        sys.exit(1)

    tree = model.tree_

    sample_features = None
    data_houses = os.path.join('..', 'data', 'houses.csv')

    df = pd.read_csv(data_houses)
    cols = ['size', 'nb_rooms']
    if all(c in df.columns for c in cols):
        sample_features = df[cols].iloc[0].values.tolist()
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 2:
            sample_features = df[numeric[:2]].iloc[0].values.tolist()

    classes = None
    if is_classifier and hasattr(model, 'classes_'):
        classes = list(model.classes_)

    print('Using sample features:', sample_features)
    c_src = generate_c(tree, sample_features, classes=classes, is_classifier=is_classifier)
    with open(C_FILE, 'w') as f:
        f.write(c_src)
    print(f'Wrote C source to {C_FILE}')

    if not compile_c(C_FILE, EXE):
        print('Compilation failed; please run the printed command manually.')
        sys.exit(1)

    out = run_exe(EXE)
    if out is None:
        print('Could not run compiled binary.')
        sys.exit(1)

    print('Binary output:', out)

if __name__ == '__main__':
    main()
