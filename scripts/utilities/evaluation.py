import numpy as np
from scipy.optimize import minimize
from .irt import one_parameter_irt
from .data_processing import read_responses_from_csv

def estimate_student_ability(data, store, student_id, beta_params, theta_guess):
    student_responses = data.get(student_id, {})
    selected_data = {}
    for index in store:
        if index < len(student_responses):
            selected_data[index] = student_responses[index]

    def negative_log_likelihood(theta):
        log_likelihood = 0
        lambda_reg = 0.1
        regularization_term = lambda_reg * theta**2
        for q_id, correct in selected_data.items():
            beta = beta_params[q_id]
            predicted_prob = one_parameter_irt(theta, beta)
            log_likelihood += correct * np.log(predicted_prob) + (1 - correct) * np.log(1 - predicted_prob)
        return -log_likelihood + regularization_term

    result = minimize(negative_log_likelihood, theta_guess, method="L-BFGS-B")
    estimated_theta = result.x[0]
    return estimated_theta

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    return rmse
