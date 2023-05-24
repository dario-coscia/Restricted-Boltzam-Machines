"""Simple Tests"""

import numpy as np
import pandas as pd
from rbm import BinaryRestrictedBoltzmannMachine
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE
from utils import fetch_mnist_data

# hyper parameters
n_components_list = [16, 36, 64, 100]
rng = 111
train_data = 1000

# methods
method = {'PCA' : PCA,
          'K-PCA' : KernelPCA,
          'ISOMAP' : Isomap,
          't-SNE' : TSNE,
          'RBM' : BinaryRestrictedBoltzmannMachine,
         }

# extra hyper parameters methods
pca_params = [{'random_state' : rng}]
isomap_params = [{'random_state' : rng}]
tsne_params = [{'random_state' : rng}]
kpca_params = [{'random_state' : rng, 'kernel' : 'rbf', 'gamma' : 0.01, 'fit_inverse_transform' : True},
               {'random_state' : rng, 'kernel' : 'rbf', 'gamma' : 0.1, 'fit_inverse_transform' : True},
               {'random_state' : rng, 'kernel' : 'rbf', 'gamma' : 1, 'fit_inverse_transform' : True},
               {'random_state' : rng, 'kernel' : 'poly', 'gamma' : 1, 'coef0' : 1, 'degree' : 1, 'fit_inverse_transform' : True},
               {'random_state' : rng, 'kernel' : 'poly', 'gamma' : 1, 'coef0' : 1, 'degree' : 2, 'fit_inverse_transform' : True},
               {'random_state' : rng, 'kernel' : 'poly', 'gamma' : 1, 'coef0' : 1, 'degree' : 3, 'fit_inverse_transform' : True}]
rbm_params = [{'random_state' : rng, 'n_iter' : 100, 'verbose' : True}]
method_params = {'PCA' : pca_params,
                 'K-PCA' : kpca_params,
                 'ISOMAP' : isomap_params,
                 't-SNE' : tsne_params,
                 'RBM' : rbm_params,
                 }

# Download dataset
X, y, X_t, y_t = fetch_mnist_data(seed=rng, train_data=train_data)

# Training models for different components
columns=['method', 'number components', 'error']
save_data = []
for n_components in n_components_list:
    # Training a model
    for model_name, model_class in method.items():
        # extract hyper-parameters
        hyper_params = method_params[model_name]
        # for each hyper parameter build a model
        for param in hyper_params:
            # build model and fit
            reducer = model_class(n_components=n_components, **param)
            err = make_pipeline_model(reducer, X, y, X_t, y_t)
            # perform analysis
            # 0. assign model specific names
            if model_name == 'RBM':
                name = str(model_name)
            elif model_name == 'PCA':
                name = str(model_name)
            elif model_name == 'K-PCA':
                kernel_type = param['kernel']
                if kernel_type == 'rbf':
                    kernel_type = 'Gaussian'
                    gamma = param['gamma']
                    name = str(model_name) + '_' + str(kernel_type) + '_' + str(gamma)
                elif kernel_type == 'poly':
                    kernel_type = 'Polynomial'
                    gamma = param['degree']
                    name = str(model_name) + '_' + str(kernel_type) + '_' + str(gamma)
            else:
                raise TypeError(f'something wrong with names... got {name}')
            # printing to see how it is going
            save_data.append([name, n_components, err])
            print(f'Test {name} for {n_components} components completed')
        
# save csv
df = pd.DataFrame(save_data, columns=columns)
df.to_csv('reconstruction_results.csv')