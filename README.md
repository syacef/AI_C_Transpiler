# AI_C_Transpiler

A small collection of scripts that train simple scikit-learn models (linear regression, logistic regression, decision trees)
and "transpile" them to plain C source code plus a compiled executable. The transpilers generate straightforward, self-contained
C code (prediction function + a small main) and compile it with GCC.

## Highlights

- Train small models with the provided training scripts.
- Transpile trained models into readable C source files.
- Compile the generated C using GCC to produce a standalone executable that prints predictions.

## Repository layout

Files and folders you'll use frequently:

- `requirements.txt` — Python dependencies (use `pip install -r requirements.txt`).
- `data/` — example CSV data used by the training/transpile scripts (e.g. `diabetes.csv`, `houses.csv`).
- `transpiler_linear_regression/`
  - `train_linear.py` — train a linear regression and save it as `model.joblib`.
  - `transpile_linear_model.py` — generate `model.c` and compile to `model`.
- `transpiler_logstic_regression/` (note the repository folder name contains a small typo: `logstic`)
  - `train_logistic.py` — train a sklearn LogisticRegression saved to `model.joblib`.
  - `transpile_logistic_model.py` — generate `model_logistic.c` and compile to `model_logistic`.
- `transpiler_decision_tree/`
  - `train_decision_tree.py` — train a DecisionTreeClassifier or DecisionTreeRegressor saved to `model_dt.joblib`.
  - `transpile_decision_tree.py` — generate `model_dt.c` and compile to `model_dt`.

## Requirements

Install dependencies (recommended in a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

requirements.txt contains pinned versions used in this project (numpy, pandas, scikit-learn, joblib).

You also need a C compiler (GCC) available in your PATH for the transpiled C to be compiled.

## Quick usage

Note: the transpiler scripts expect the trained model file to be present in the same folder they run from (or follow the constants inside the script). The examples below run each pair of train + transpile from its folder so sample data lookup works as implemented.

Linear regression (example):

```bash
cd transpiler_linear_regression
python train_linear.py        # creates model.joblib
python transpile_linear_model.py
# Outputs: model.c (C source) and compiled executable `model`.
./model                      # runs the compiled binary and prints a prediction
```

Logistic regression (binary only):

```bash
cd transpiler_logstic_regression
python train_logistic.py     # creates model.joblib
python transpile_logistic_model.py
# Outputs: model_logistic.c and executable `model_logistic`.
./model_logistic
```

Decision tree (classifier or regressor):

```bash
cd transpiler_decision_tree
python train_decision_tree.py   # creates model_dt.joblib
python transpile_decision_tree.py
# Outputs: model_dt.c and executable `model_dt`.
./model_dt
```

## What the transpilers generate

- C source files (e.g. `model.c`, `model_logistic.c`, `model_dt.c`) containing a small prediction routine and `main()` which
  initializes a sample feature vector and prints the predicted value or class.
- The generated code uses plain C and standard math functions (e.g. `expf` for logistic). The transpilers call `gcc -O3 -std=c99 ...`.

Constants and output names (from the scripts):

- Linear: writes `model.c` and compiles to executable `model`.
- Logistic: writes `model_logistic.c` and compiles to `model_logistic` (links `-lm`).
- Decision tree: writes `model_dt.c` and compiles to `model_dt`.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.