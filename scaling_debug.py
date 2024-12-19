import numpy as np
from scipy.linalg import svd
import pyomo.environ as pyo
import idaes.core.util.scaling as iscale
jac, nlp = iscale.get_jacobian(blk)
variables = nlp.get_pyomo_variables()
constraints = nlp.get_pyomo_equality_constraints()
print("Badly scaled variables:")
for i in iscale.extreme_jacobian_columns(
        jac=jac, nlp=nlp, large=1E5, small=1E-5):
    print(f"    {i[0]:.2e}, [{i[1]}]")
print("\n\n" + "Badly scaled constraints:")
for i in iscale.extreme_jacobian_rows(
        jac=jac, nlp=nlp, large=1E5, small=1E-5):
    print(f"    {i[0]:.2e}, [{i[1]}]")
# print(f"Jacobian Condition Number: {iscale.jacobian_cond(jac=jac):.2e}")
# if not hasattr(m.fs, "obj"):
#     m.fs.obj = pyo.Objective(expr=0)
n_sv = 10
u, s, vT = svd(jac.todense(), full_matrices=False)

print("\n" + f"Spectral condition number: {s[0]/s[-1]:.3e}")
# Reorder singular values and vectors so that the singular
# values are from least to greatest
u = np.flip(u[:, -n_sv:], axis=1)
s = np.flip(s[-n_sv:], axis=0)
vT = np.flip(vT[-n_sv:, :], axis=0)
v = vT.transpose()
print("\n" + f"Smallest singular value: {s[0]}")
print("\n" + "Variables in smallest singular vector:")
for i in np.where(abs(v[:, 0]) > 0.2)[0]:
    print(str(i) + ": " + variables[i].name)
print("\n" + "Constraints in smallest singular vector:")
for i in np.where(abs(u[:, 0]) > 0.2)[0]:
    print(str(i) + ": " + constraints[i].name)