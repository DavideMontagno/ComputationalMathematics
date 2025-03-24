# Computational Mathematics for Learning and Data Analysis
**_Project for the course of Computational Mathematics for Learning and Data Analysis @ University of Pisa_**

<p align="center">
  <img width=150px src="https://www.plan4res.eu/wp-content/uploads/2018/02/University-of-Pisa-Italy.png"/>
</p>

## Code structure
```
├── README.md
├── assignment.pdf
├── projects_guidelines.pdf
├── CM_Project_ML-3.pdf
├── datasets
│   ├── ML_CUP
│   │   └── ML-CUP20-TR.csv
│   └── Szeged
│       ├── weatherHistory_original.csv
│       └── weatherHistory_preprocessed.csv
├── requirements.txt
└── src
    ├── dataset_analysis.ipynb
    ├── functions.py
    ├── m1.py
    ├── m1_utilities.py
    ├── m2.py
    ├── pytorch_baseline.py
    └── weight_initializations.py
```

### Modules description
The actual code is all contained in the `src` directory and here we provide a description of the various modules:

#### M1
- `m1.py`: contains the implementation of the fully-connected feed-forward neural network and its methods (for training, validation, etc.)
- `m1_utilities.py`: contains some utility functions related to M1
- `functions.py`: contains the implementation of the functions needed by M1, in particular:
  - class `Function`: represents a function object characterized by a name and a pointer to a function executing the function itself
  - class `DerivableFunction`: inherits from `Function` and represents a derivable function, which has also a pointer to a function executing the derivative
  - activation functions and their derivatives
  - loss functions and their derivative
  - regularization functions and their derivative
- `weight_initializations.py`: contains the implementation of different weight initializations for M1
- `pytorch_baseline.py`: contains the code to run a training using a PyTorch implementation to compare it with our implementation

#### M2
- `m2.py`: contains the implementation of the model M2 plus the relative algorithms and experiments
- `dataset_analysis.ipynb`: contains some analysis and preprocessing of the Szeged dataset used for M2

### Instructions
- `m1_utilities.py`: The `main` function contains a basic demo to 
  - run GridSearch (implemented in `grid_search` function)
  - select the best model/models relative to SGD/HB/NAG (implemented in `pick_best_models` function) 
  - show the error gaps (implemented in `show_err_and_plot` function) w.r.t. an optimum value
- `m2.py`: The `main` function contains a basic demo to
  - test the implementation of least square using as metrics:
    - minimum relative approximation error (between Ax and b)
    - approximation error bound (with gradient) between the computed and the actual solution
  - test the implementation of least square with L2 reg (both with and without noise in the input data) using as metrics:
    - minimum relative approximation error (between Ax and b)
    - approximation error bound (with gradient) between the computed and the actual solution

#### Quick run
m1_utilities:
```
cd src
python m1_utilities.py
```
m2:
```
cd src
python m2.py
```
