import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import gpflow
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import NMF

from .graphnn import models as molan_model
from .graphnn import training
from .graphnn import mol2graph

import torch
from torch_geometric.data import DataLoader

from collections import OrderedDict

from rdkit import Chem

class DNN(keras.Model):
    _n_layers = 1
    _layer_size = 16
    batch_size = 32
    learning_rate = 0.0001
    epochs = 500
    seed = 9700

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.generate_fcn()

    def generate_fcn(self):
        self.pipeline = []

        for i, layer in enumerate(range(self.n_layers)):
            self.pipeline.append(layers.BatchNormalization())
            self.pipeline.append(layers.Dense(self.layer_size, activation='relu'))
        
        self.pipeline.append(layers.BatchNormalization())
        self.pipeline.append(layers.Dense(1, activation='linear'))


    @property
    def n_layers(self):
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value):
        self._n_layers = value
        self.generate_fcn()

    @property
    def layer_size(self):
        return self._layer_size

    @layer_size.setter
    def layer_size(self, value):
        self._layer_size = value
        self.generate_fcn()

    def call(self, inputs):
        x = inputs

        for layer in self.pipeline:
            x = layer(x)
            
        return x

    def fit(self, x_train, y_train, **kwargs):
        tf.random.set_seed(self.seed)

        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        super().build(input_shape=x_train.shape)
        super().compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
        super().fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, **kwargs)

class DNN_Mordred(DNN):
    _n_layers = 4
    _layer_size = 256
    batch_size = 256
    learning_rate = 0.01
    epochs = 1000

class DNN_ECFP(DNN):
    _n_layers = 1
    _layer_size = 2048
    batch_size = 512
    learning_rate = 0.001
    epochs = 1000

class RF:
    seed = 9700
    n_estimators = 4096
    max_depth = 32
    min_samples_split = 2
    min_samples_leaf = 1

    def fit(self, x_train, y_train):
        np.random.seed(self.seed)

        self.estimator = RandomForestRegressor(n_estimators = self.n_estimators,
                                                max_depth = self.max_depth,
                                                min_samples_split = self.min_samples_split,
                                                min_samples_leaf = self.min_samples_leaf, 
                                                n_jobs=-1)

        self.estimator.fit(x_train, y_train.ravel())

        return self

    def predict(self, x):
        if self.estimator is None:
            raise NotImplementedError()

        return self.estimator.predict(x).reshape(-1,1)
    
    def save_weights(self, fn):
        with open(fn, 'wb') as file:
            pickle.dump(self.estimator, file)

    def load_weights(self, fn):
        with open(fn, 'rb') as file:
            self.estimator = pickle.load(file)

class RF_NMF_ECFP(RF):
    def fit(self, x_train, y_train, seed=9700):
        self.estimator = make_pipeline(
            NMF(n_components=12, solver='mu', init='random', max_iter=500, random_state=0, alpha=.1, l1_ratio=.5),
            RandomForestRegressor(min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_depth = self.max_depth,
                                    n_estimators=self.n_estimators,
                                    n_jobs=-1)
        )

        self.estimator.fit(x_train, y_train.ravel())

        return self


class GP:
    def fit(self, x_train, y_train):
        kernels = []
        i = 0

        for no, (x, reducer, k) in enumerate(zip(x_train,
                                self.rf_feature_selectors,
                                self.rf_feature_reduce_to)):

            indices = (-reducer.estimator.feature_importances_).argsort()[:k]
            x_train[no] = x[:,indices]
            kernels.append(gpflow.kernels.RBF(active_dims=i+np.arange(k)))

            i += k

        x_train = np.hstack(x_train)
        kernel = gpflow.kernels.Sum(kernels)

        self.model = gpflow.models.GPR(data=(x_train.astype(np.float64), y_train.astype(np.float64)), kernel=kernel,
                                        mean_function=None)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(lambda: -self.model.log_marginal_likelihood(), self.model.trainable_variables,
            options={'maxiter': 500})

    def predict(self, x_in):
        for no, (x, reducer, k) in enumerate(zip(x_in,
                                self.rf_feature_selectors,
                                self.rf_feature_reduce_to)):

            indices = (-reducer.estimator.feature_importances_).argsort()[:k]
            x_in[no] = x[:,indices]

        x = np.hstack(x_in)
        return self.model.predict_y(x.astype(np.float64))[0]

    def save_weights(self, fn):
        checkpoint = tf.train.Checkpoint(a=self.model)
        manager = tf.train.CheckpointManager(checkpoint, fn, max_to_keep=9999)
        
        manager.save()

class SN_Mordred:
    batch_size = 256
    learning_rate = 0.004663515283240011
    epochs = 1000
    seed = 9700

    def __init__(self, input_shape=None):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        x = layers.Input(shape=input_shape)

        body = layers.BatchNormalization()(x)
        body = layers.Dense(128, activation='relu')(body)
        body = layers.BatchNormalization()(body)
        body = layers.Dense(128, activation='relu')(body)
        body = layers.BatchNormalization()(body)
        body = layers.Dense(128, activation='relu')(body)
        body = layers.BatchNormalization()(body)

        prediction = layers.Dense(1, activation='linear', name='prediction')(body)

        selection = layers.Dense(16, activation='relu')(body)
        selection = layers.BatchNormalization()(selection)
        selection = layers.Dense(1, activation='sigmoid', name='selection')(selection)

        selection_out = layers.Concatenate(axis=1, name='selection_head')([prediction, selection])
        auxiliary_out = layers.Dense(1, activation='linear', name='auxiliary_head')(body)

        self.model = tf.keras.models.Model(inputs=x, outputs=[selection_out, auxiliary_out, body])

    def fit(self, x_train, y_train, **kwargs):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        def coverage(y_true, y_pred):
            return K.mean(K.round(y_pred[:,1]))
        
        def empirical_risk(y_true, y_pred):
            loss = (y_true[:,0] - y_pred[:,0])**2

            mse = K.mean(loss * y_pred[:,1])
            
            emp_risk_num = mse
            emp_risk_denom = K.mean(y_pred[:,1]) #K.mean(K.round(y_pred[:,1]))
            
            return emp_risk_num / emp_risk_denom

        def selective_loss(y_true, y_pred):
            emp_risk = empirical_risk(y_true, y_pred)
            cov = K.mean(y_pred[:,1])
            
            lamda = 32 #converge later
            loss = emp_risk + (lamda * K.maximum(self.c_coverage-cov,0)**2)

            return loss

        self.model.compile(optimizer=adam, loss=[selective_loss, 'mse'], loss_weights=[0.5, 0.5],
            metrics={'selection_head': [selective_loss, empirical_risk, coverage]})
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, **kwargs)

    def predict(self, *args):
        return self.model.predict(*args)

    def save_weights(self, fn):
        return self.model.save_weights(fn)

    def load_weights(self, fn):
        return self.model.load_weights(fn)

class GCN:
    # Some code here taken directly from MOLAN
    seed = 9700
    conv_n_layers = 5
    conv_base_size = 64
    conv_ratio = 1.25
    conv_batchnorm = True
    conv_act = 'relu'
    emb_dim = 100
    emb_set2set = False
    emb_act = 'softmax'
    mlp_layers = 2
    mlp_dim_ratio = 0.5
    mlp_dropout = 0.15306049825909776
    mlp_act = 'relu'
    mlp_batchnorm = True
    residual = False
    learning_rate = 0.008117123009364938
    batch_size = 64
    epochs = 500
    node_dim = mol2graph.n_atom_features()
    edge_dim = mol2graph.n_bond_features()

    def fit(self, x_train, y_train):
        torch.manual_seed(self.seed)

        hparams = OrderedDict([('conv_n_layers', self.conv_n_layers), ('conv_base_size', self.conv_base_size),
                        ('conv_ratio', self.conv_ratio), ('conv_batchnorm', self.conv_batchnorm),
                        ('conv_act', self.conv_act), ('emb_dim', self.emb_dim),
                        ('emb_set2set', self.emb_set2set), ('emb_act', self.emb_act),
                        ('mlp_layers', self.mlp_layers), ('mlp_dim_ratio', self.mlp_dim_ratio),
                        ('mlp_dropout', self.mlp_dropout), ('mlp_act', self.mlp_act),
                        ('mlp_batchnorm', self.mlp_batchnorm), ('residual', self.residual)])

        hparams['lr'] = self.learning_rate
        hparams['batch_size'] = self.batch_size
        hparams['model'] = 'GCN'

        x_train = [mol2graph.mol2torchdata(Chem.MolFromSmiles(smile)) for smile in x_train.flatten()]

        for data, y in zip(x_train, y_train):
            data.y = torch.tensor(y, dtype=torch.float)

        loader = DataLoader(x_train, batch_size=self.batch_size,
            shuffle=False, drop_last=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = molan_model.GCN(hparams, self.node_dim, self.edge_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        mode = 'min',
                        factor = 0.5,
                        patience = 20,
                        verbose = True)

        for i in range(self.epochs):
            print('Step %d/%d' % (i+1, self.epochs))
            training.train_step(self.model, loader, optimizer, scheduler, self.device)

    def predict(self, x_in):
        # should drop_last=False
        x_in = [mol2graph.mol2torchdata(Chem.MolFromSmiles(smile)) for smile in x_in.flatten()]

        loader = DataLoader(x_in, batch_size=1,
            shuffle=False, drop_last=False)

        results = []

        with torch.no_grad():
            self.model.eval()
            
            for data in loader:
                data = data.to(self.device)
                output = self.model(data)
                results.extend(output.cpu().numpy())

        return np.array(results).reshape(-1,1)

    def save_weights(self, fn):
        torch.save(self.model.state_dict(), fn)

    def load_weights(self, fn):
        hparams = OrderedDict([('conv_n_layers', self.conv_n_layers), ('conv_base_size', self.conv_base_size),
                        ('conv_ratio', self.conv_ratio), ('conv_batchnorm', self.conv_batchnorm),
                        ('conv_act', self.conv_act), ('emb_dim', self.emb_dim),
                        ('emb_set2set', self.emb_set2set), ('emb_act', self.emb_act),
                        ('mlp_layers', self.mlp_layers), ('mlp_dim_ratio', self.mlp_dim_ratio),
                        ('mlp_dropout', self.mlp_dropout), ('mlp_act', self.mlp_act),
                        ('mlp_batchnorm', self.mlp_batchnorm), ('residual', self.residual)])

        hparams['lr'] = self.learning_rate
        hparams['batch_size'] = self.batch_size
        hparams['model'] = 'GCN'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = molan_model.GCN(hparams, self.node_dim, self.edge_dim)
        self.model.load_state_dict(torch.load(fn))