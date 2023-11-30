
from collections import defaultdict
import numpy as np
import pandas as pd

class SpacedRepetitionModelHLR:
    MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes
    MAX_HALF_LIFE = 274.0             # 9 months
    LN2 = np.log(2)

    def __init__(self, feature_columns, lrate=.001, hlwt=.01, l2wt=.1, sigma=1.):
        self.feature_columns = feature_columns
        self.weights = np.random.randn(len(feature_columns)) * 0.01
        self.fcounts = defaultdict(float)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma
    
    def pclip(self, p):
        return np.clip(p, 0.0001, 0.9999)
    
    def hclip(self, h):
        return np.clip(h, self.MIN_HALF_LIFE, self.MAX_HALF_LIFE)
    
    def halflife(self, X):
        dp = X.dot(self.weights)
        return self.hclip(2 ** dp)
    
    def find_p(self, h_hat, t):
        p_hat = 2 ** (-t / h_hat)
        return self.pclip(p_hat)
    
    def losses(self, X, t, p, h):
        h_hat = self.halflife(X)
        p_hat = self.find_p(h_hat, t)
        slp = (p - p_hat) ** 2
        slh = (h - h_hat) ** 2

        return slp, slh, p_hat, h_hat
        
    def calculate_total_loss(self, X, p, t, h):
        slp, slh, _, _ = self.losses(X, t, p, h)
        total_slp = np.sum(slp)
        total_slh = np.sum(slh)
        total_l2 = np.sum(self.weights ** 2)
        total_loss = total_slp + self.hlwt * total_slh + self.l2wt * total_l2
        return total_loss

    def train(self, dataframe_x, dataframe_y, max_iter=2000, tolerance=1e-7, print_iter=1000):
        X = dataframe_x[self.feature_columns].values
        p = dataframe_y.values.flatten()
        t = dataframe_x['t'].values
        h = dataframe_x['h'].values
        
        with open('training_hlr.txt', 'w') as file:

            for iteration in range(max_iter):
                h_hat = self.halflife(X)
                p_hat = self.find_p(h_hat, t)

                for k in range(len(self.feature_columns)):
                    # Calcula la derivada para la característica k
                    dlp_dw_k = 2. * (p_hat - p) * self.LN2 * p_hat * (-t / h_hat) * X[:, k]
                    dlh_dw_k = 2. * (h_hat - h) * self.LN2 * h_hat * X[:, k]

                    # Suma sobre todas las instancias para obtener el gradiente total para la característica k
                    gradient_k = np.sum(dlp_dw_k) + self.hlwt * np.sum(dlh_dw_k)

                    # Tasa de aprendizaje ajustada
                    # rate = ( 1 /( 1 + p ) ) * self.lrate / np.sqrt(1 + self.fcounts[k])
                    rate = self.lrate / np.sqrt(1 + self.fcounts[k])

                    # Actualiza el peso
                    self.weights[k] -= rate * gradient_k
                    self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2

                    # Incrementa el conteo de la característica
                    self.fcounts[k] += 1

                if iteration % print_iter == 0:                   
                    total_loss = self.calculate_total_loss(X, p, t, h)
                    print(f"Iteration {iteration}, Total Loss: {total_loss}")
                    file.write(f"Iteration {iteration}:\n")
                    file.write(f"Total Loss: {total_loss}\n")
                    file.write(f"Learning Rate: {rate}\n")
                    file.write(f"F counts: { self.fcounts }\n")
                    file.write(f"Weights: {self.weights}\n")
                    file.write("\n")  # Espacio adicional para separar las entradas


                # Implementar una verificación de convergencia basada en `tolerance`
