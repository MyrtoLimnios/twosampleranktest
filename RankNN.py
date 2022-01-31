# Initial implementation from https://github.com/liyinxiao/LambdaRankNN
# Modified for the two-sample procedure as detailed in the companion paper

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense, Input, Subtract
from tensorflow.keras.models import Model
import numpy as np
import math
from sklearn.metrics import dcg_score

class RankerNN(object):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        """
        Parameters
        ----------
        input_size : integer
            Number of input features.
        hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
            The ith element represents the number of neurons in the ith
            hidden layer.
        activation : tuple, length = n_layers - 2, default ('relu',)
            The ith element represents activation function in the ith
            hidden layer.
        solver : {'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', adamax},
        default 'adam'
            The solver for weight optimization.
            - 'adam' refers to a stochastic gradient-based optimizer proposed
              by Kingma, Diederik, and Jimmy Ba
        """
        if len(hidden_layer_sizes) != len(activation):
            raise ValueError('hidden_layer_sizes and activation should have the same size.')
        self.model = self._build_model(input_size, hidden_layer_sizes, activation)
        self.model.compile(optimizer=solver, loss="binary_crossentropy")
        print('Warning myrtolimnios: modified pairwise step')

    @staticmethod
    def _build_model(input_shape, hidden_layer_sizes, activation):
        """
        Build Keras Ranker NN model (Ranknet / LambdaRank NN).
        """
        # Neural network structure
        hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            hidden_layers.append(
                Dense(hidden_layer_sizes[i], activation=activation[i], name=str(activation[i]) + '_layer' + str(i)))
        h0 = Dense(1, activation='linear', name='Identity_layer')
        input1 = Input(shape=(input_shape,), name='Input_layer1')
        input2 = Input(shape=(input_shape,), name='Input_layer2')
        x1 = input1
        x2 = input2
        for i in range(len(hidden_layer_sizes)):
            x1 = hidden_layers[i](x1)
            x2 = hidden_layers[i](x2)
        x1 = h0(x1)
        x2 = h0(x2)
        # Subtract layer
        subtracted = Subtract(name='Subtract_layer')([x1, x2])
        # sigmoid
        out = Activation('sigmoid', name='Activation_layer')(subtracted)
        # build model
        model = Model(inputs=[input1, input2], outputs=out)
        return model

    @staticmethod
    def _CalcDCG(labels):
        sumdcg = 0.0
        for i in range(len(labels)):
            rel = labels[i]
            if rel != 0:
                sumdcg += ((2 ** rel) - 1) / math.log2(i + 2)
        return sumdcg



    def _transform_pairwise(self, X, y):
        return None, None, None, None

    def fit(self, X, y, batch_size=None, epochs=1, verbose=1, validation_split=0.0):
        """Transform data and fit model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        """
        X1_trans, X2_trans, y_trans, weight = self._transform_pairwise(X, y)
        self.model.fit([X1_trans, X2_trans], y_trans, sample_weight=weight, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, validation_split=validation_split)
        #self.evaluate(X, y)

    def predict(self, X):
        """Predict output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        Returns
        -------
        y_pred: array, shape (n_samples,)
            Model prediction.
        """
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])

        return ranker_output([X])[0].ravel()

    def evaluate(self, X, y, eval_at=None):
        """Predict and evaluate ndcg@eval_at.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        eval_at: integer
            The rank postion to evaluate DCG.
        Returns
        -------
        dcg@eval_at: float
        """
        y_pred = self.predict(X)
        #tmp = np.array(np.hstack([y.reshape(-1, 1), y_pred.reshape(-1, 1)]))
        #tmp = tmp[np.argsort(-tmp[:, 1])]
        #y_sorted = tmp[:, 0]
        print(dcg_score(y, y_pred))


class RankNetNN(RankerNN):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        super(RankNetNN, self).__init__(input_size, hidden_layer_sizes, activation, solver)

    def transform_pairwise(self, X, y):
        """Transform data into ranknet pairs with balanced labels for
        binary classification.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        Returns
        -------
        X1_trans : array, shape (k, n_feaures)
            Features of pair 1
        X2_trans : array, shape (k, n_feaures)
            Features of pair 2
        weight: array, shape (k, n_faetures)
            Sample weight lambda.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {0, 1}
        """
        X1 = []
        X2 = []
        weight = []
        Y = []
        ### we have to limit pos_idx to the first sample ie to the score 1 // Suppose only one query
        for pos_idx in np.where(y == 1)[0]:
            # for pos_idx in range(len(rel_list)):
            for all_idx in range(len(y)):
                #X1.append(X[pos_idx])
                #X2.append(X[all_idx])
                # balanced class
                # if y[all_idx] == 1:
                    #weight.append(1)
                    #Y.append(1)
                #else:
                    # weight.append(1)
                    # Y.append(0)
                if 1 != (-1) ** (pos_idx + all_idx):
                    X1.append(X[pos_idx])
                    X2.append(X[all_idx])
                    weight.append(1)
                    Y.append(1)
                else:
                    X1.append(X[all_idx])
                    X2.append(X[pos_idx])
                    weight.append(1)
                    Y.append(0)
        return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)

    def _transform_pairwise(self, X, y):
        """Transform data into ranknet pairs with balanced labels for
        binary classification.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        Returns
        -------
        X1_trans : array, shape (k, n_feaures)
            Features of pair 1
        X2_trans : array, shape (k, n_feaures)
            Features of pair 2
        weight: array, shape (k, n_faetures)
            Sample weight lambda.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {0, 1}
        """
        X1 = []
        X2 = []
        weight = []
        Y = []
        ### we have to limit pos_idx to the first sample ie to the score 1 // Suppose only one query
        for pos_idx in np.where(y == 1)[0]:
            # for pos_idx in range(len(rel_list)):
            for all_idx in range(len(y)):
                X1.append(X[pos_idx])
                X2.append(X[all_idx])
                if y[all_idx] == 1:
                    weight.append(1)
                    Y.append(1)
                else:
                    weight.append(1)
                    Y.append(0)
        return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)


class LambdaRankNN(RankerNN):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        super(LambdaRankNN, self).__init__(input_size, hidden_layer_sizes, activation, solver)

    def transform_pairwise(self, X, y, qid):
        """Transform data into lambdarank pairs with balanced labels
        for binary classification.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        Returns
        -------
        X1_trans : array, shape (k, n_feaures)
            Features of pair 1
        X2_trans : array, shape (k, n_feaures)
            Features of pair 2
        weight: array, shape (k, n_faetures)
            Sample weight lambda.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {0, 1}
        """
        X1 = []
        X2 = []
        weight = []
        Y = []
        IDCG = np.sum([1.0 / math.log2(k + 2) for k in range(np.sum(y))])
        ### we have to limit pos_idx to the first sample ie to the score 1 // Suppose only one query
        for pos_idx in np.where(y == 1.0)[0]:
            # for pos_idx in range(len(rel_list)):
            for all_idx in range(len(y)):
                # calculate lambda
                pos_loginv = 1.0 / math.log2(pos_idx + 2)
                all_loginv = 1.0 / math.log2(all_idx + 2)
                pos_label = y[pos_idx]
                all_label = y[all_idx]
                original = ((1 << pos_label) - 1) * pos_loginv + ((1 << all_label) - 1) * all_loginv
                changed = ((1 << all_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * all_loginv
                # We do not use the Normalized DCG (nDCG) as we consider only one query
                # IDCG: ideal  DCG, used to normalise the DCG
                delta = (original - changed) * IDCG
                if delta < 0:
                    delta = -delta
                # balanced class
                if 1 != (-1) ** (pos_idx + all_idx):
                    X1.append(X[pos_idx])
                    X2.append(X[all_idx])
                    weight.append(delta)
                    Y.append(1)
                else:
                    X1.append(X[all_idx])
                    X2.append(X[pos_idx])
                    weight.append(delta)
                    Y.append(0)
        return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)

    def _transform_pairwise(self, X, y):
        """Transform data into ranknet pairs with balanced labels for
        binary classification.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        Returns
        -------
        X1_trans : array, shape (k, n_feaures)
            Features of pair 1
        X2_trans : array, shape (k, n_feaures)
            Features of pair 2
        weight: array, shape (k, n_faetures)
            Sample weight lambda.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {0, 1}
        """
        X1 = []
        X2 = []
        weight = []
        Y = []
        ### we have to limit pos_idx to the first sample ie to the score 1 // Suppose only one query
        for pos_idx in np.where(y == 1)[0]:
            # for pos_idx in range(len(rel_list)):
            for all_idx in range(len(y)):
                X1.append(X[pos_idx])
                X2.append(X[all_idx])
                if y[all_idx] == 1:
                    weight.append(1)
                    Y.append(1)
                else:
                    weight.append(1)
                    Y.append(0)

        IDCG = np.sum([1.0 / math.log2(k + 2) for k in range(int(np.sum(y)))])
        ### we have to limit pos_idx to the first sample ie to the score 1 // Suppose only one query
        for pos_idx in np.where(y == 1.0)[0]:
            # for pos_idx in range(len(rel_list)):
            for all_idx in range(len(y)):
                # calculate lambda
                pos_loginv = 1.0 / math.log2(pos_idx + 2)
                all_loginv = 1.0 / math.log2(all_idx + 2)
                pos_label = 1
                all_label = y[all_idx]
                original = ((1 << pos_label) - 1) * pos_loginv + ((1 << all_label) - 1) * all_loginv
                changed = ((1 << all_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * all_loginv
                # We do not use the Normalized DCG (nDCG) as we consider only one query
                # IDCG: ideal  DCG, used to normalise the DCG
                delta = (original - changed) * IDCG
                if delta < 0:
                    delta = -delta
                X1.append(X[pos_idx])
                X2.append(X[all_idx])
                if y[all_idx] == 1:
                    weight.append(delta)
                    Y.append(1)
                else:
                    weight.append(delta)
                    Y.append(0)

        return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)
