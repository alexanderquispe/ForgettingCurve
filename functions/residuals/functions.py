
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

class hlr_adagrad:
    
    def __init__(self, feature_columns, alpha=0.01, lambda_=0.1, eta=0.001, epsilon=0.001):
        
        """
        Initializes the Half Life Regression Model with adaptive learning rate (AdaGrad).

        :param feature_columns: List of feature names.
        :param alpha: Weight for the half-life term in the loss function.
        :param lambda_: Regularization parameter.
        :param eta: Initial learning rate for AdaGrad.
        :param epsilon: Small value to avoid division by zero in AdaGrad.
        """        

        self.theta = np.random.randn( len( feature_columns ) ) * 0.01
        self.feature_columns = feature_columns
        self.alpha = alpha
        self.lambda_ = lambda_
        self.eta = eta
        self.epsilon = epsilon

    def _find_h_hat(self, X):
        min_h = 15.0 / ( 24 * 60 ) # 15 Mins
        max_h = 274 # 9 months
        theta_x = X.dot(self.theta)
        h_hat = 2 ** theta_x
        h_hat = np.clip( h_hat, min_h, max_h )
        return h_hat

    def _find_p_hat(self, h_hat, Delta):
        p_hat = 2 ** -Delta / h_hat
        p_hat = np.clip( p_hat, 0.0001, 0.9999 )
        return p_hat

    def _cost_function(self, X, p, Delta):
        h_hat = self._find_h_hat(X)
        p_hat = self._find_p_hat(h_hat, Delta)
        h = -Delta / np.log2(p + self.epsilon)
        total_cost = np.mean((p - p_hat) ** 2 + self.alpha * (h - h_hat) ** 2 + 
                             self.lambda_ * np.sum(self.theta ** 2))
        # total_cost += self.lambda_ * np.sum(self.theta ** 2)
        return total_cost

    def train(self, dataframe_x, dataframe_y, max_iter=1000000, tolerance=1e-7, print_iter=1000):
        X = dataframe_x[self.feature_columns].values
        p = dataframe_y.values.flatten()
        Delta = dataframe_x['t'].values
        grad_accumulation = np.zeros_like(self.theta)
        cost_history = []

        for iteration in range(max_iter):
            h_hat = self._find_h_hat(X)
            p_hat = self._find_p_hat(h_hat, Delta)
            # grad_theta = -2 * X.T.dot((p - p_hat) * np.log(2) * p_hat * (2 ** (-Delta / h_hat)) + 
                                      # self.alpha * (h_hat + Delta / np.log2(p)) * np.log(2) * h_hat) + 2 * self.lambda_ * self.theta
                
            grad_theta = -2 * X.T.dot( ( p_hat - p ) * np.log( 2 ) * p_hat * (  -Delta / h_hat  ) + 2 * self.alpha * ( h_hat + Delta / np.log2( p ) ) * np.log( 2 ) * h_hat ) + 2 * self.lambda_ * self.theta
            grad_accumulation += grad_theta ** 2

            adjusted_eta = self.eta / (np.sqrt(grad_accumulation) + self.epsilon)
            self.theta -= adjusted_eta * grad_theta / len(X)
            cost = self._cost_function(X, p, Delta)
            cost_history.append(cost)

            if iteration % print_iter == 0:
                print(f"Iteration {iteration}, Loss: {cost}")

            # Check for convergence
            if iteration > 0 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
                print(f"Convergence reached at iteration {iteration}, Loss {cost}")
                break

        return cost_history

    def predict(self, row):
        x = np.array([row[feature] for feature in self.feature_columns])
        h_hat = self._find_h_hat(x[np.newaxis, :])[0]
        p_hat = self._find_p_hat(h_hat, row['t'])
        
        return p_hat, h_hat
    
    
    

# class hlr_gd:
    
#     def __init__(self, feature_columns, alpha=0.01, lambda_=0.1, eta=0.001, epsilon=0.001):
        
#         """
#         Initializes the Half Life Regression Model with Basic Gradient Descent.

#         :param feature_columns: List of feature names.
#         :param alpha: Weight for the half-life term in the loss function.
#         :param lambda_: Regularization parameter.
#         :param eta: Initial learning rate for AdaGrad.
#         :param epsilon: Small value to avoid division by zero in AdaGrad.
#         """        

#         self.theta = np.random.randn( len( feature_columns ) ) * 0.01
#         self.feature_columns = feature_columns
#         self.alpha = alpha
#         self.lambda_ = lambda_
#         self.eta = eta
#         self.epsilon = epsilon

#     def _find_h_hat(self, X):
#         min_h = 15.0 / ( 24 * 60 ) # 15 Mins
#         max_h = 274 # 9 months
#         theta_x = X.dot(self.theta)
#         h_hat = 2 ** theta_x
#         h_hat = np.clip( h_hat, min_h, max_h )
#         return h_hat

#     def _find_p_hat(self, h_hat, Delta):
#         p_hat = 2 ** -Delta / h_hat
#         p_hat = np.clip( p_hat, 0.0001, 0.9999 )
#         return p_hat

#     def _cost_function(self, X, p, Delta):
#         h_hat = self._find_h_hat(X)
#         p_hat = self._find_p_hat(h_hat, Delta)
#         h = -Delta / np.log2(p + self.epsilon)
#         total_cost = np.mean((p - p_hat) ** 2 + self.alpha * (h - h_hat) ** 2)
#         total_cost += self.lambda_ * np.sum(self.theta ** 2)
#         return total_cost

#     def train(self, dataframe_x, dataframe_y, max_iter=1000000, tolerance=1e-7, print_iter=1000):
#         X = dataframe_x[self.feature_columns].values
#         p = dataframe_y.values.flatten()
#         Delta = dataframe_x['t'].values
#         cost_history = []

#         for iteration in range(max_iter):
#             h_hat = self._find_h_hat(X)
#             p_hat = self._find_p_hat(h_hat, Delta)
#             grad_theta = -2 * X.T.dot((p - p_hat) * np.log(2) * p_hat * (2 ** (-Delta / h_hat)) + 
#                                       self.alpha * (h_hat + Delta / np.log2(p)) * np.log(2) * h_hat) + 2 * self.lambda_ * self.theta

#             # Actualización de los pesos usando descenso de gradiente
#             self.theta -= self.eta * grad_theta / len(X)

#             cost = self._cost_function(X, p, Delta)
#             cost_history.append(cost)

#             if iteration % print_iter == 0:
#                 print(f"Iteration {iteration}, Loss: {cost}")

#             # Check for convergence
#             if iteration > 0 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
#                 print(f"Convergence reached at iteration {iteration}, Loss {cost}")
#                 break

#         return cost_history

#     def predict(self, row):
#         x = np.array([row[feature] for feature in self.feature_columns])
#         h_hat = self._find_h_hat(x[np.newaxis, :])[0]
#         p_hat = self._find_p_hat(h_hat, row['t'])
        
#         return p_hat, h_hat
    
    

# import numpy as np

# class hlr_minibatches:
#     def __init__(self, feature_columns, alpha=0.01, lambda_=0.1, eta=0.001, epsilon=0.001):
        
#         """
#         Initializes the Half Life Regression Model with Minibatches.

#         :param feature_columns: List of feature names.
#         :param alpha: Weight for the half-life term in the loss function.
#         :param lambda_: Regularization parameter.
#         :param eta: Initial learning rate for AdaGrad.
#         :param epsilon: Small value to avoid division by zero in AdaGrad.
#         """        

#         self.theta = np.random.randn( len( feature_columns ) ) * 0.01
#         self.feature_columns = feature_columns
#         self.alpha = alpha
#         self.lambda_ = lambda_
#         self.eta = eta
#         self.epsilon = epsilon

#     def _find_h_hat(self, X):
#         min_h = 15.0 / ( 24 * 60 ) # 15 Mins
#         max_h = 274 # 9 months
#         theta_x = X.dot(self.theta)
#         h_hat = 2 ** theta_x
#         h_hat = np.clip( h_hat, min_h, max_h )
#         return h_hat

#     def _find_p_hat(self, h_hat, Delta):
#         p_hat = 2 ** -Delta / h_hat
#         p_hat = np.clip( p_hat, 0.0001, 0.9999 )
#         return p_hat

#     def _cost_function(self, X, p, Delta):
#         h_hat = self._find_h_hat(X)
#         p_hat = self._find_p_hat(h_hat, Delta)
#         h = -Delta / np.log2(p + self.epsilon)
#         total_cost = np.mean((p - p_hat) ** 2 + self.alpha * (h - h_hat) ** 2)
#         total_cost += self.lambda_ * np.sum(self.theta ** 2)
#         return total_cost
#     def train(self, dataframe_x, dataframe_y, max_iter=1000000, minibatch_size=32, tolerance=1e-7, print_iter=1000):
#         X = dataframe_x[self.feature_columns].values
#         p = dataframe_y.values.flatten()
#         Delta = dataframe_x['t'].values
#         num_samples = len(X)
#         cost_history = []

#         for iteration in range(max_iter):
#             # Crear índices aleatorios para minibatches
#             indices = np.random.permutation(num_samples)
#             for i in range(0, num_samples, minibatch_size):
#                 batch_indices = indices[i:i + minibatch_size]
#                 X_batch = X[batch_indices]
#                 p_batch = p[batch_indices]
#                 Delta_batch = Delta[batch_indices]

#                 # Calcular gradientes para el minibatch
#                 h_hat = self._find_h_hat(X_batch)
#                 p_hat = self._find_p_hat(h_hat, Delta_batch)
#                 grad_theta = -2 * X_batch.T.dot((p_batch - p_hat) * np.log(2) * p_hat * (2 ** (-Delta_batch / h_hat)) + 
#                                                 self.alpha * (h_hat + Delta_batch / np.log2(p_batch)) * np.log(2) * h_hat) + 2 * self.lambda_ * self.theta

#                 # Actualizar los pesos
#                 self.theta -= self.eta * grad_theta / len(X_batch)

#             # Calcular la pérdida después de procesar todos los minibatches
#             cost = self._cost_function(X, p, Delta)
#             cost_history.append(cost)

#             if iteration % print_iter == 0:
#                 print(f"Iteration {iteration}, Loss: {cost}")

#             # Check for convergence
#             if iteration > 0 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
#                 print(f"Convergence reached at iteration {iteration}, Loss {cost}")
#                 break

#         return cost_history

#     def predict(self, row):
#         x = np.array([row[feature] for feature in self.feature_columns])
#         h_hat = self._find_h_hat(x[np.newaxis, :])[0]
#         p_hat = self._find_p_hat(h_hat, row['t'])
        
#         return p_hat, h_hat

    

# class hlr_adagrad_adapted:
#     def __init__(self, feature_columns, alpha=0.01, lambda_=0.1, eta=0.001, epsilon=0.001):
#         self.theta = np.random.randn(len(feature_columns)) * 0.01
#         self.feature_columns = feature_columns
#         self.alpha = alpha
#         self.lambda_ = lambda_
#         self.eta = eta
#         self.epsilon = epsilon
#         self.grad_accumulation = np.zeros_like(self.theta)
#         self.feature_count = np.zeros_like(self.theta)

#     def _find_h_hat(self, X):
#         min_h = 15.0 / (24 * 60)  # 15 Mins
#         max_h = 274  # 9 months
#         theta_x = X.dot(self.theta)
#         h_hat = 2 ** theta_x
#         h_hat = np.clip(h_hat, min_h, max_h)
#         return h_hat

#     def _find_p_hat(self, h_hat, Delta):
#         p_hat = 2 ** -Delta / h_hat
#         p_hat = np.clip(p_hat, 0.0001, 0.9999)
#         return p_hat

#     def _cost_function(self, X, p, Delta):
#         h_hat = self._find_h_hat(X)
#         p_hat = self._find_p_hat(h_hat, Delta)
#         h = -Delta / np.log2(p + self.epsilon)
#         total_cost = np.mean((p - p_hat) ** 2 + self.alpha * (h - h_hat) ** 2)
#         total_cost += self.lambda_ * np.sum(self.theta ** 2)
#         return total_cost

#     def train(self, dataframe_x, dataframe_y, max_iter=1000000, tolerance=1e-7, print_iter=1000):
#         X = dataframe_x[self.feature_columns].values
#         p = dataframe_y.values.flatten()
#         Delta = dataframe_x['t'].values

#         for iteration in range(max_iter):
#             h_hat = self._find_h_hat(X)
#             p_hat = self._find_p_hat(h_hat, Delta)

#             # Update feature count
#             self.feature_count += np.sum(X != 0, axis=0)

#             # Calculate gradient
#             grad_theta = -2 * X.T.dot((p - p_hat) * np.log(2) * p_hat * (2 ** (-Delta / h_hat)) + 
#                                       self.alpha * (h_hat + Delta / np.log2(p)) * np.log(2) * h_hat) + 2 * self.lambda_ * self.theta

#             # Update gradient accumulation
#             self.grad_accumulation += grad_theta ** 2

#             # Adjust learning rate
#             adjusted_eta = self.eta / (np.sqrt(self.grad_accumulation / self.feature_count) + self.epsilon)

#             # Update parameters
#             self.theta -= adjusted_eta * grad_theta

#             # Calculate cost
#             cost = self._cost_function(X, p, Delta)
#             if iteration % print_iter == 0:
#                 print(f"Iteration {iteration}, Loss: {cost}")

#             # Check for convergence
#             if iteration > 0 and np.abs(cost - self._cost_function(X, p, Delta)) < tolerance:
#                 print(f"Convergence reached at iteration {iteration}, Loss {cost}")
#                 break

#         return cost

#     def predict(self, row):
#         x = np.array([row[feature] for feature in self.feature_columns])
#         h_hat = self._find_h_hat(x[np.newaxis, :])[0]
#         p_hat = self._find_p_hat(h_hat, row['t'])

#         return p_hat, h_hat
