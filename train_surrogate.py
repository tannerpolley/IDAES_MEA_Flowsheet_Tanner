# Import statements
import os
import numpy as np
import pandas as pd

# Import Pyomo libraries
from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    value,
    Var,
    Constraint,
    Set,
    Objective,
    maximize,
)
from pyomo.common.timing import TicTocTimer

# Import IDAES libraries
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.alamopy import AlamoTrainer, AlamoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_parity,
    surrogate_residual,
)
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core import FlowsheetBlock
from idaes.core.util.convergence.convergence_base import _run_ipopt_with_stats

# Import Auto-reformer training data
np.set_printoptions(precision=6, suppress=True)

csv_data = pd.read_csv("surrogate_training_data_absorber.csv")  # 2800 data points
data = csv_data.sample(n=30000)  # randomly sample points for training/validation

input_data = data.iloc[:, 1:-1]
output_data = data.iloc[:, -1]

# Define labels, and split training and validation data
input_labels = input_data.columns
output_labels = [output_data.name]

n_data = data[input_labels[0]].size
data_training, data_validation = split_training_validation(
    data, 0.8, seed=n_data
)  # seed=100

print(input_labels)
print(output_labels)

# column_mean = data_training.mean()
# data_training = data_training - column_mean
# data_validation = data_validation - column_mean

# capture long output (not required to use surrogate API)
from io import StringIO
import sys

# stream = StringIO()
# oldstdout = sys.stdout
# sys.stdout = stream

# Create ALAMO trainer object
trainer = AlamoTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# Set ALAMO options
trainer.config.modeler = 2
trainer.config.constant = True
trainer.config.linfcns = True
trainer.config.expfcns = True
trainer.config.logfcns = True
trainer.config.sinfcns = False
trainer.config.cosfcns = False
trainer.config.multi2power = [1, 2, 3, 4]
trainer.config.multi3power = [1, 2, 3]
trainer.config.monomialpower = [2, 3, 4, 5]
trainer.config.ratiopower = [1, 2, 3, 4]
trainer.config.maxterms = [30]
trainer.config.filename = os.path.join(os.getcwd(), "alamo_run.alm")
trainer.config.overwrite_files = True
# trainer.config.builder = False
# trainer.config.solvemip = True
# trainer.config.xfactor = [1/3, 1, 1, 10, 1e-4, 1e-5]
trainer.config.xfactor = [1/3, 10, 1, 300, 1e-4, 1e-5]
# trainer.config.xscaling = True

# Train surrogate (calls ALAMO through IDAES ALAMOPy wrapper)
has_alamo = True
try:
    success, alm_surr, msg = trainer.train_surrogate()
except FileNotFoundError as err:
    if "Could not find ALAMO" in str(err):
        print("ALAMO not found. You must install ALAMO to use this notebook")
        has_alamo = False
    else:
        raise

if has_alamo:
    # save model to JSON
    model = alm_surr.save_to_file("alamo_surrogate_absorber.json", overwrite=True)

    # create callable surrogate object

    surrogate_expressions = trainer._results["Model"]
    input_labels = trainer._input_labels
    output_labels = trainer._output_labels
    xmin = [0.05, 6.5, 8.923, 360, 5e-5, 3e-5]
    xmax = [0.6, 8.5, 11.225, 405, 5e-4, 5.5e-5]
    input_bounds = {
        input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
    }

    alm_surr = AlamoSurrogate(
        surrogate_expressions, input_labels, output_labels, input_bounds
    )

    # # revert back to normal output capture
    # sys.stdout = oldstdout

    # # display first 50 lines and last 50 lines of output
    # celloutput = stream.getvalue().split("\n")
    # for line in celloutput[:50]:
    #     print(line)
    # print(".")
    # print(".")
    # print(".")
    # for line in celloutput[-50:]:
    #     print(line)

    if has_alamo:
        # visualize with IDAES surrogate plotting tools
        surrogate_scatter2D(alm_surr, data_training, filename="alamo_train_scatter2D.pdf")
        surrogate_parity(alm_surr, data_training, filename="alamo_train_parity.pdf")
        surrogate_residual(alm_surr, data_training, filename="alamo_train_residual.pdf")

    # if has_alamo:
    #     # visualize with IDAES surrogate plotting tools
    #     surrogate_scatter2D(alm_surr, data_validation, filename="alamo_val_scatter2D.pdf")
    #     surrogate_parity(alm_surr, data_validation, filename="alamo_val_parity.pdf")
    #     surrogate_residual(alm_surr, data_validation, filename="alamo_val_residual.pdf")