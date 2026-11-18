""" Implements custom transformer for the special kind of grouped datasets
that neuralxc is working with.

A typical dataset looks like this:
[{'spec1': features,'spec2' : features}, {'spec1': features, 'spec3': features}]
where the outer list runs over independent systems.
"""
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC

from ..formatter import atomic_shape, system_shape

TorchModule = torch.nn.Module

def convert_torch_wrapper(func):
    def wrapped_func(X, *args, **kwargs):
        # Recursively strip away any nested tracking tuples introduced by JIT tracing or sklearn wrappers
        while isinstance(X, tuple) and len(X) > 0:
            X = X[0]
            
        cleaned_args = []
        for arg in args:
            while isinstance(arg, tuple) and len(arg) > 0:
                arg = arg[0]
            cleaned_args.append(arg)
        args = tuple(cleaned_args)

        made_tensor = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            made_tensor = True
            
        Y = func(X, *args, **kwargs)
        
        while isinstance(Y, tuple) and len(Y) > 0:
            Y = Y[0]
            
        return Y
    return wrapped_func


class GroupedTransformer(ABC):
    """ Abstract base class, grouped transformer extend the functionality
    of sklearn Transformers to neuralxc specific grouped data. Further, they
    implement a get_gradient method.
    """
    def __init__(self, *args, **kwargs):
        TorchModule.__init__(self)
        self.is_fit = False
        self.is_wrapped = False
        super().__init__(*args, **kwargs)

    def __sklearn_is_fitted__(self):
        """Communicates fitting status directly to modern scikit-learn pipelines."""
        return getattr(self, "is_fit", False)

    def set_output(self, *, transform=None):
        """Disables modern scikit-learn's automatic tuple metadata tracking framework."""
        return self

    def transform(self, X, y=None, **fit_params):
        if not hasattr(self, 'is_wrapped'):
            self.is_wrapped = False
        if fit_params.get('wrap_torch', True) and hasattr(self, '_spec_dict'):
            for spec in self._spec_dict:
                trafo = self._spec_dict[spec]
                if hasattr(trafo, 'torch_transform') and not getattr(trafo, 'is_wrapped', False):
                    trafo.torch_transform = convert_torch_wrapper(trafo.torch_transform)
                    trafo.is_wrapped = True            
        #if fit_params.get('wrap_torch', True) and hasattr(self, '_spec_dict'):
        #    for spec in self._spec_dict:
        #        trafo = self._spec_dict[spec]
        #        trafo.torch_transform = convert_torch_wrapper(trafo.torch_transform)
                
        was_tuple = False
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
            was_tuple = True

        made_list = False
        if not isinstance(X, list):
            X = [X]
            made_list = True

        results = []
        for x in X:
            if isinstance(x, dict):
                results_dict = {}
                for spec in x:
                    data_matrix = x[spec]
                    original_shape = None

                    if hasattr(data_matrix, 'ndim') and data_matrix.ndim > 2:
                        original_shape = data_matrix.shape
                        data_matrix = data_matrix.reshape(-1, original_shape[-1])

                    res = self._spec_dict[spec].transform(data_matrix)

                    if isinstance(res, tuple) and len(res) > 0:
                        res = res[0]

                    if original_shape is not None:
                        new_feature_dim = res.shape[-1]
                        target_shape = list(original_shape[:-1]) + [new_feature_dim]
                        res = res.reshape(target_shape)

                    results_dict[spec] = res
                results.append(results_dict)
            else:
                # Restored to standard pristine array mapping block
                results.append(system_shape(self.torch_transform(atomic_shape(x)), x.shape[-2]))        
        #for x in X:
        #    if isinstance(x, dict):
        #        results_dict = {}
        #        for spec in x:
        #            data_matrix = x[spec]
        #            original_shape = None

        #            # Flatten 3D spatial matrices to 2D for scikit-learn
        #            if hasattr(data_matrix, 'ndim') and data_matrix.ndim > 2:
        #                original_shape = data_matrix.shape
        #                data_matrix = data_matrix.reshape(-1, original_shape[-1])

        #            # Safely run through the modern native scikit-learn transformer step
        #            res = self._spec_dict[spec].transform(data_matrix)

        #            if isinstance(res, tuple) and len(res) > 0:
        #                res = res[0]

        #            # Reconstruct the original 3D geometry layout if necessary
        #            if original_shape is not None:
        #                new_feature_dim = res.shape[-1]
        #                target_shape = list(original_shape[:-1]) + [new_feature_dim]
        #                res = res.reshape(target_shape)

        #            results_dict[spec] = res
        #        results.append(results_dict)
        #    else:
        #        # --- FIXED: Recursively unpack multi-layered tracking containers safely ---
        #        while isinstance(x, tuple) and len(x) > 0:
        #            x = x[0]
        #        results.append(system_shape(self.torch_transform(atomic_shape(x)), x.shape[-2]))

        if made_list:
            results = results[0]
        if was_tuple:
            return results, y
        else:
            return results

    def fit(self, X, y=None):
        if self.is_fit:
            return self
        else:
            self.is_fit = True
            if isinstance(X, tuple):
                X = X[0]

            if isinstance(X, list):
                super_X = {}
                for x in X:
                    for spec in x:
                        if not spec in super_X:
                            super_X[spec] = []
                        super_X[spec].append(atomic_shape(x[spec]))
                for spec in super_X:
                    super_X[spec] = np.concatenate(super_X[spec])
            else:
                super_X = X

            if isinstance(super_X, dict):
                self._spec_dict = {}
                for spec in super_X:
                    self._spec_dict[spec] = type(self)(*self._initargs, **self.get_kwargs())
                    self._spec_dict[spec].__dict__.update(self.get_params())
                    self._spec_dict[spec].is_fit = False
                    mask = ~np.all(atomic_shape(super_X[spec]) == 0, axis=-1)
                    self._spec_dict[spec].fit(self._before_fit(atomic_shape(super_X[spec])[mask]))
                return self
            else:
                return super().fit(atomic_shape(super_X))
    #def fit(self, X, y=None):
    #    if self.is_fit:
    #        return self
    #    else:
    #        self.is_fit = True
    #        if isinstance(X, tuple):
    #            X = X[0]

    #        if isinstance(X, list):
    #            super_X = {}
    #            for x in X:
    #                for spec in x:
    #                    if not spec in super_X:
    #                        super_X[spec] = []
    #                    super_X[spec].append(atomic_shape(x[spec]))
    #            for spec in super_X:
    #                self._spec_dict[spec] = type(self)(*self._initargs, **self.get_kwargs())
    #                self._spec_dict[spec].__dict__.update(self.get_params())
    #                self._spec_dict[spec].is_fit = False
    #                mask = ~np.all(atomic_shape(super_X[spec]) == 0, axis=-1)
    #                self._spec_dict[spec].fit(self._before_fit(atomic_shape(super_X[spec])[mask]))
    #                #super_X[spec] = np.concatenate(super_X[spec])
    #        else:
    #            super_X = X

    #        if isinstance(super_X, dict):
    #            self._spec_dict = {}
    #            for spec in super_X:
    #                self._spec_dict[spec] = type(self)(*self._initargs, **self.get_kwargs())
    #                self._spec_dict[spec].__dict__.update(self.get_params())
    #                self._spec_dict[spec].is_fit = False
    #                mask = ~np.all(atomic_shape(super_X[spec]) == 0, axis=-1)
    #                self._spec_dict[spec].fit(self._before_fit(atomic_shape(super_X[spec])[mask]))                
    #            #for spec in super_X:
    #            #    self._spec_dict[spec] = type(self)(*self._initargs, **self.get_kwargs())
    #            #    self._spec_dict[spec].__dict__.update(self.get_params())
    #            #    mask = ~np.all(atomic_shape(super_X[spec]) == 0, axis=-1)
    #            #    self._spec_dict[spec].fit(self._before_fit(atomic_shape(super_X[spec])[mask]))
    #            return self
    #        else:
    #            return super().fit(atomic_shape(super_X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def forward(self, X):
        return self.transform(X, wrap_torch=False)

    def to_torch(self):
        pass


class GroupedVarianceThreshold(GroupedTransformer, VarianceThreshold, TorchModule):
    def __init__(self, threshold=0.0):
        TorchModule.__init__(self)
        self._before_fit = identity  
        self._initargs = []
        self.treshold = threshold
        super().__init__(**self.get_kwargs())

    def get_kwargs(self):
        return dict(threshold=self.treshold)

    def transform(self, X, y=None, **fit_params):
        return GroupedTransformer.transform(self, X, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        return GroupedTransformer.fit_transform(self, X, y, **fit_params)

    def torch_transform(self, X):
        while isinstance(X, tuple) and len(X) > 0:
            X = X[0]
        X_shape = X.size()
        if not len(X_shape) == 2:
            X = X.view(-1, X_shape[-1])

        support = self.get_support()
        while isinstance(support, tuple) and len(support) > 0:
            support = support[0]

        support = torch.from_numpy(np.asarray(support, dtype=bool))
        ## --- FIXED: Bypass scikit-learn property lookup during forward calls ---
        #support = self.__dict__.get('_support_mask', None)
        #if support is None:
        #    support = self.__dict__.get('support_', None)
        #if support is None:
        #    # Fallback only if private dictionaries aren't populated yet
        #    support = self.get_support()
        #    
        #while isinstance(support, tuple) and len(support) > 0:
        #    support = support[0]
        #    
        #support = torch.from_numpy(support).bool()
        return X[:, support]


class GroupedStandardScaler(GroupedTransformer, StandardScaler, TorchModule):
    def __init__(self, threshold=0.0):
        TorchModule.__init__(self)
        self._before_fit = identity  
        self._initargs = []
        self.threshold = threshold
        self._initkwargs = {}
        super().__init__()

    def get_kwargs(self):
        return {}

    def transform(self, X, y=None, **fit_params):
        return GroupedTransformer.transform(self, X, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        return GroupedTransformer.fit_transform(self, X, y, **fit_params)

    def torch_transform(self, X):
        while isinstance(X, tuple) and len(X) > 0:
            X = X[0]
        X_shape = X.size()
        if not len(X_shape) == 2:
            X = X.view(-1, X_shape[-1])

        mean = self.__dict__.get('mean_', getattr(self, 'mean_', None))
        while isinstance(mean, tuple) and len(mean) > 0:
            mean = mean[0]
        if mean is None:
            mean = np.zeros(X_shape[-1], dtype=np.float64)

        var = self.__dict__.get('var_', getattr(self, 'var_', None))
        while isinstance(var, tuple) and len(var) > 0:
            var = var[0]
        if var is None:
            var = np.ones(X_shape[-1], dtype=np.float64)

        mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float64)).to(dtype=X.dtype, device=X.device)
        var_t = torch.from_numpy(np.asarray(var, dtype=np.float64)).to(dtype=X.dtype, device=X.device)
        return (X - mean_t) / torch.sqrt(var_t)
        ## --- FIXED: Extract raw arrays directly from internal dictionary memory ---
        #mean = self.__dict__.get('mean_', None)
        #while isinstance(mean, tuple) and len(mean) > 0:
        #    mean = mean[0]
        #    
        #var = self.__dict__.get('var_', None)
        #while isinstance(var, tuple) and len(var) > 0:
        #    var = var[0]
        #    
        #X = (X - torch.from_numpy(mean)) / torch.sqrt(torch.from_numpy(var))
        #return X


def identity(x):
    return x
