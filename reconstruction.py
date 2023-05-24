"""Simple Tests"""

import numpy as np
import pandas as pd
import sys
from rbm import BinaryRestrictedBoltzmannMachine
from sklearn.decomposition import PCA, KernelPCA
from utils import fetch_mnist_data, plot_samples, reconstruction_error

# hyper parameters
n_components_list = [16, 64, 100]
rng = 111
train_data = 2000
gibbs_steps=10
noise = float(sys.argv[1])

# methods
method = {'PCA' : PCA,
          'RBM' : BinaryRestrictedBoltzmannMachine,
         }

# extra hyper parameters methods
pca_params = [{'random_state' : rng}]
rbm_params = [{'random_state' : rng, 'n_iter' : 100, 'verbose' : True}]
method_params = {'PCA' : pca_params,
                 'RBM' : rbm_params,
                 }

# Download dataset
X, _, X_t, _ = fetch_mnist_data(seed=rng, train_data=train_data)
plot_samples(samples=X_t, numb_panels=64, title='MNIST Original', save_title='original_mnist')
X_noise = X_t + np.random.standard_normal(size=X_t.shape) * noise
plot_samples(samples=X_noise, numb_panels=64, title=f'MNIST Original + Noise {noise}%', save_title='noise_mnist')

# Training models for different components
columns=['method', 'number components', 'MSE', 'std MSE']
save_data = []
for n_components in n_components_list:
    # Training a model
    for model_name, model_class in method.items():
        # extract hyper-parameters
        hyper_params = method_params[model_name]
        # for each hyper parameter build a model
        for param in hyper_params:
            # build model and fit
            model = model_class(n_components=n_components, **param)
            model.fit(X)
            # perform analysis
            # 0. assign model specific names
            if model_name == 'RBM':
                name = str(model_name)
            elif model_name == 'PCA':
                name = str(model_name)
            else:
                raise TypeError(f'something wrong with names... got {name}')
            # 1. plotting originals  + reconstruction + hidden states
            X_reduced = model.transform(X_noise)
            # if there is noise
            if noise > 0 and model_name=='RBM':
                model._gibbs_steps = gibbs_steps
            X_reconstructed = model.inverse_transform(X_reduced)
            plot_samples(samples=X_reconstructed, numb_panels=64, title=f'{name} reconstruction', save_title=f'reconstruction_{n_components}_{name}')
            # 2. reconstruction error
            err, std = reconstruction_error(X_t, X_reconstructed)
            save_data.append([name, n_components, err, std])

            # printing to see how it is going
            print(f'Test {name} for {n_components} components completed')
        
# save csv
df = pd.DataFrame(save_data, columns=columns)
df.to_csv(f'reconstruction_results_{noise}.csv')