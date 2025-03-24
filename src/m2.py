import math
import notebook.nbextensions
import numpy as np
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_dataset():
    """
    Reads the dataset split into inputs and targets
    :return: tha inputs and the targets
    """
    A = pd.read_csv('../datasets/Szeged/weatherHistory_preprocessed.csv', index_col=0)
    # A = A.sample(frac=1.)
    temperatures = A.pop('Temperature (C)')
    A, temperatures = A.to_numpy(), temperatures.to_numpy()
    temperatures /= np.linalg.norm(temperatures)
    return A, temperatures


def check_sensitivity(A: np.ndarray, noise: np.ndarray) -> bool:
    """
    Checks the sensitivity, i.e. ||sigma_i - sigma_i_noise|| <= ||E||
    where sigma_i = Singular Values of A
          sigma_i_noise = Singular Values of A+E (where E is some noise added to A)
    """
    noise_norm = linalg.norm(noise)
    A_E = A + noise
    _, singular_vals, _ = linalg.svd(A, full_matrices=False)
    _, singular_vals_noise, _ = linalg.svd(A_E, full_matrices=False)
    for i in range(len(singular_vals)):
        diff = linalg.norm(singular_vals[i] - singular_vals_noise[i])
        if diff > noise_norm:
            return False
    return True


def solve_least_squares(b, A=None, U=None, singular_vals=None, Vt=None):
    """
    Solves the least squares problem (mix_x ||Ax-b||)
    :param b: target vector
    :param A: input matrix
    :param U: U matrix of the SVD (A = U @ S @ Vt), containing left singular vectors
    :param singular_vals: array containing the singular values
    :param Vt: V matrix of the SVD (A = U @ S @ Vt) transposed, containing the right singular vectors
    :return: the solution vector x
    """
    if A is None and (U is None or singular_vals is None or Vt is None):
        raise AttributeError("The function requires that either A is passed, or its decomposition is passed (U, singular_vals, Vt)")

    if A is not None:
        if U is not None or singular_vals is not None or Vt is not None:
            print(f"WARNING: A has been passed, but also U, singular_vals and Vt. The last 3 are going to be ignored and recomputed from A")
        U, singular_vals, Vt = linalg.svd(A, full_matrices=False)
    y = [U[:, i].T @ b / singular_vals[i] for i in range(len(singular_vals))]
    x = Vt.T @ y
    return x


def solve_least_squares_tikhonov(A, b, alpha):
    """
    Solves the least squares problem with Tikhonov regularization
    :param A: input matrix
    :param b: target vector
    :param alpha: regularization coefficient
    :return: the solution vector x
    """
    AtA = A.T @ A
    I = np.identity(len(AtA))
    x = np.linalg.inv(AtA + alpha ** 2 * I) @ A.T @ b
    return x


def relative_cond(A):
    """
    Computes the relative condition number of a matrix
    :param A: matrix of which to compute the relative condition number
    :return: the relative condition number of A
    """
    _, singular_values, _ = linalg.svd(A, full_matrices=False)
    largest, smallest = singular_values[0], singular_values[-1]
    return largest / smallest


def test_l2_reg():
    """
    Solves the least squares problem with L2 reg with different values of the reg coefficient (alpha)
    """
    # read dataset
    A, b = read_dataset()

    # create perturbation to be added to A and check sensitivity
    noise = 0.4 * np.random.normal(size=A.shape)
    print("For all i, ||sigma_i - sigma_noise_i|| <= ||noise_matrix|| verified: ", check_sensitivity(A, noise))

    # add noise to A and split the data
    A_noise = A + noise
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.3, shuffle=True, random_state=3242)
    A_train_noise, A_test_noise, b_train_noise, b_test_noise = train_test_split(A_noise, b, test_size=0.3, shuffle=True, random_state=3242)
    reg_coeffs = (.9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)

    # initialize variables to save the minimum errors
    min_relat_approx_err = math.inf
    min_relat_approx_err_alpha = reg_coeffs[0]
    min_err_bound_grad = math.inf
    min_err_alpha_grad = reg_coeffs[0]
    min_relat_approx_err_noise = math.inf
    min_relat_approx_err_noise_alpha = reg_coeffs[0]
    min_err_bound_grad_noise = math.inf
    min_err_noise_alpha_grad = reg_coeffs[0]

    # try all the regularization coefficients
    for alpha in reg_coeffs:
        # solve the least squares with L2 reg, both with and without noise
        x = solve_least_squares_tikhonov(A_train, b_train, alpha)
        x_noise = solve_least_squares_tikhonov(A_train_noise, b_train_noise, alpha)

        # compute relative approximation error of the solution
        relat_approx_err = np.linalg.norm(A_test @ x - b_test) / np.linalg.norm(b_test)

        # compute the bound on the difference between the computed solution and the actual one (checking the gradient)
        # ∇f(x) = A.TAx - A^tb
        grad_f = (A_test.T @ A_test @ x) - (A_test.T @ b_test)
        # norm ∇f(x)
        norm_grad_f = np.linalg.norm(grad_f)
        # condition number (A^TA) == condition number(A)^2
        cond_AtA = relative_cond(A_test.T @ A_test)  # relative_cond(A_test)**2
        # k(A.TA) * (||ATAx - A^Tb|| / || A.T @ b || )
        bound_err_x_xapprox_grad = cond_AtA * norm_grad_f / np.linalg.norm(A_test.T @ b_test)

        # compute relative approximation error of the solution of the noisy problem
        relat_approx_err_noise = np.linalg.norm(A_test_noise @ x_noise - b_test_noise) / np.linalg.norm(b_test_noise)

        # compute the bound on the difference between the computed solution of the noisy problem and the actual one (checking the gradient)
        bound_err_x_xapprox_grad_noise = \
            relative_cond(A_test_noise.T @ A_test_noise) * \
            np.linalg.norm((A_test_noise.T @ A_test_noise @ x_noise) - (A_test_noise.T @ b_test_noise)) / \
            np.linalg.norm(A_test_noise.T @ b_test_noise)

        print("\nAlpha: ", alpha)
        print("Relative approximation error bound with gradient: ", bound_err_x_xapprox_grad)
        print("Relative approximation error bound with gradient (with noise): ", bound_err_x_xapprox_grad)

        if relat_approx_err < min_relat_approx_err:
            min_relat_approx_err = relat_approx_err
            min_relat_approx_err_alpha = alpha
        if bound_err_x_xapprox_grad < min_err_bound_grad:
            min_err_bound_grad = bound_err_x_xapprox_grad
            min_err_alpha_grad = alpha
        if relat_approx_err_noise < min_relat_approx_err_noise:
            min_relat_approx_err_noise = relat_approx_err_noise
            min_relat_approx_err_noise_alpha = alpha
        if bound_err_x_xapprox_grad_noise < min_err_bound_grad_noise:
            min_err_bound_grad_noise = bound_err_x_xapprox_grad_noise
            min_err_noise_alpha_grad = alpha

    print("\n\nMinimum relative approximation error (between Ax and b): ", min_relat_approx_err, "- Alpha:", min_relat_approx_err_alpha)
    print("Minimum approximation error bound with gradient (between true solution and x): ", min_err_bound_grad, "- Alpha:", min_err_alpha_grad)
    print("Minimum relative approximation error (between Ax and b) with noise: ", min_relat_approx_err_noise, "- Alpha:",
          min_relat_approx_err_noise_alpha)
    print("Minimum approximation error bound with gradient (between true solution and x) with noise: ", min_err_bound_grad_noise,
          "- Alpha:", min_err_noise_alpha_grad)


if __name__ == '__main__':
    A, b = read_dataset()
    matrix_rank = np.linalg.matrix_rank(A)
    print(f"A's rank: {matrix_rank} (number of columns: {A.shape[1]})")

    # SVD decomposition
    U, singular_vals, Vt = linalg.svd(A, full_matrices=False)
    S = np.diag(singular_vals)  # create S diagonal matrix from the list of singular values
    print("Norm (A - USV.T): ", np.linalg.norm(A - (U @ S @ Vt)))

    """
    Solving least square problem
    LEAST SQUARE: mix_x ||Ax-b||
    """
    print("----------------------------------------------\nLEAST SQUARE\n")
    U, singular_vals, Vt = linalg.svd(A, full_matrices=False)
    x = solve_least_squares(A=A, b=b)   # solve L.S. and get solution vector x
    relat_err_bound_grad = relative_cond(A) ** 2 * np.linalg.norm(A.T @ A @ x - A.T @ b) / np.linalg.norm(A.T @ b)
    print("Approximation error bound (with gradient) between the computed and the actual solution: ", relat_err_bound_grad)
    print("Error: ", np.linalg.norm(b - A @ x), "(rank: ", len(singular_vals), ")")
    print("Relative error: ", np.divide(np.linalg.norm(A @ x - b), np.linalg.norm(b)))
    print("b is close to Ax: ", np.allclose(np.dot(A, x), b, atol=5e-3))
    print("max |Ax - b|: ", np.max(np.abs(np.dot(A, x) - b)))

    # least squares with SVD and L2 regularization
    print("----------------------------------------------\nLEAST SQUARE with L2 regularization\n")
    test_l2_reg()
