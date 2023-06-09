"""Utils file."""

import numpy as np


def fetch_mnist_data(seed, train_data=1000, subset = None):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import minmax_scale
    # fix random state for fetching
    rng = np.random.RandomState(seed)
    index_number= rng.permutation(70000)
    # Download dataset
    index_number= rng.permutation(70000)
    mnist = fetch_openml('mnist_784', parser='auto', version=1, cache=True)
    X, y  = mnist.data.loc[index_number],mnist.target.loc[index_number]
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    if subset is not None:
        X, y = X[:subset], y[:subset]
    # Normalize (0, 1)
    X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling
    return X[:train_data], y[:train_data], X[train_data:], y[train_data:]

def plot_samples(samples, numb_panels, title, save_title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.2, 4))
    for i in range(numb_panels):
        plt.subplot(int(np.sqrt(numb_panels)), int(np.sqrt(numb_panels)), i + 1)
        plt.imshow(samples[i].reshape((int(np.sqrt(samples.shape[1])), -1)), cmap=plt.cm.gray, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(title, fontsize=12)
    #plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 1.38, 0.23)
    plt.tight_layout()
    plt.savefig(f'{save_title}.pdf')

def save_model(file_name, model):
    import pickle
    #save model
    with open(f'{file_name}.pickle', 'wb') as file:
        pickle.dump(model, file) 

def load_model(file_name, model):
    import pickle
    #load model
    with open(f'{file_name}.pickle', 'rb') as file:
        pickle.load(model, file) 

def reconstruction_error(input_, target):
    from sklearn.metrics import mean_squared_error
    err = mean_squared_error(target, input_, multioutput='raw_values')
    return float(np.mean(err)), float(np.std(err))

def make_pipeline_model(reducer, X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(max_iter=500, solver='newton-cg')
    # fit reducer + logistic
    latent = reducer.fit_transform(X_train)
    logistic.fit(latent, y_train)
    train_acc = logistic.score(latent, y_train)
    # test
    latent = reducer.transform(X_test)
    test_acc = logistic.score(latent, y_test)
    return train_acc, test_acc