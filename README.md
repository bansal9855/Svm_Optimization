# SVM Optimization using Random Search

This repository contains a Python implementation for optimizing Support Vector Machine (SVM) hyperparameters using random search. The script evaluates different kernels and regularization parameters to achieve the best classification accuracy on the Wine dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Output Format](#output-format)
- [Sample Input and Output](#sample-input-and-output)
- [License](#license)

## Installation

To run the SVM optimization script, you need to have Python 3.x installed along with the necessary libraries. You can install the required packages using pip:

```bash
pip install pandas matplotlib scikit-learn
```

## Usage

You can run the SVM optimization script directly. It will load the Wine dataset, perform the optimization, and generate a convergence graph and results CSV file.

### Command to Run:

```bash
python main.py
```

## Dataset

The script uses the Wine dataset from the `sklearn.datasets` library, which includes the following features:

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of Diluted Wines
- Proline

The target variable is the wine class (0, 1, or 2).

## Output Format

Upon execution, the script generates:

1. A convergence graph (`convergence_graph.png`) showing the best accuracy over iterations.
2. A CSV file (`SVM_Convergence_Data.csv`) containing the following columns:
   - `Iteration`: The iteration number.
   - `Best Accuracy`: The best accuracy achieved at that iteration.
   - `Best Kernel`: The kernel used for the best accuracy.
   - `Best Nu`: The regularization parameter used for the best accuracy.

### Example of the Output CSV file (`SVM_Convergence_Data.csv`):

```csv
Iteration,Best Accuracy,Best Kernel,Best Nu
0,0.9778,linear,0.5678
1,0.9778,linear,0.3456
2,0.9889,rbf,0.4567
...
```

## Sample Input and Output

### Sample Command:

To execute the script, simply run:

```bash
python svm_optimization.py
```

### Sample Output:

After running the command, you will see printed results in the console, such as:

```
Current working directory: /path/to/your/directory
Convergence graph saved at: /path/to/your/directory/convergence_graph.png
Results saved at: /path/to/your/directory/SVM_Convergence_Data.csv
Best Accuracy: 0.9889
Best SVM Parameters: Kernel=rbf, Nu=0.4567
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
```

