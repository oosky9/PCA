import numpy as np
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis:

    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
      
    def fit(self, x):
        self.pca.fit(x)

    def process_projection(self, x):
        return self.pca.transform(x)

    def process_inverse(self, x):
        return self.pca.inverse_transform(x)

    def set_eigen_vector(self, x):
        self.pca.components_ = x
    
    def set_eigen_value(self, x):
        self.pca.explained_variance_ = x
    
    def set_mean_vector(self, x):
        self.pca.mean_ = x
   
    def get_eigen_vector(self):
        return self.pca.components_

    def get_eigen_value(self):
        return self.pca.explained_variance_
    
    def get_mean_vector(self):
        return self.pca.mean_

    def get_contribution_ratio(self):
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_contribution_ratio(self):
        CR = self.get_contribution_ratio()
        
        total = 0
        CCR = []
        for i in range(len(CR)):
            total += CR[i]
            CCR.append(total)
        
        return CCR

