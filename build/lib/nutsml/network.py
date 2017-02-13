"""
.. module:: network
   :synopsis: Wrapper around other network APIs such as Lasagne or Keras
              to enable usage within nuts-flow.
              For instance, with a wrapped network one can write:
              samples >> build_batch >> network.train() >> log_loss >> Consume()
"""

import numpy as np

from nutsflow import (nut_processor, nut_sink, nut_function, Collect, Map,
                      Flatten)


@nut_processor
def TrainValNut(batches, func):
    """
    batches >> TrainValNut(func)

    Create nut to train or validate a network.

    :param iterable over batches batches: Batches to train/validate.
    :param function func: Training or validation function of network.
    :return: Result(s) of training/validation function, e.g. loss, accuracy, ...
    :rtype: float or array/tuple of floats
    """
    for batch in batches:
        yield func(*batch)


@nut_processor
def PredictNut(batches, func, flatten=True):
    """
    batches >> PredictNut(func)

    Create nut to perform network predictions.

    :param iterable over batches batches: Batches to create predictions for.
    :param function func: Prediction function
    :param bool flatten: True: flatten output. Instead of returning batch of
           predictions return individual predictions
    :return: Result(s) of prediction
    :rtype: typically array with class probabilities (softmax vector)
    """
    for batch in batches:
        pred_batch = func(*batch)
        if flatten:
            for prediction in pred_batch:
                yield prediction
        else:
            yield pred_batch


@nut_sink
def EvalNut(batches, network, metrics, predcol=None, targetcol=-1):
    """
    batches >> EvalNut(network, metrics)

    Create nut to evaluate network performance for given metrics.
    Returned when network.evaluate() is called.

    :param iterable over batches batches: Batches to evaluate
    :param nutmsml.Network network:
    :param list of functions metrics: List of functions that compute
           some metric, e.g. accuracy, F1, kappa-score.
           Each metric function must take vectors with true and
           predicted  classes/probabilities and must compute the
           metric over the entire input (not per sample/mini-batch).
    :param int|None predcol: Index of column in prediction to extract
           for evaluation. If None a single prediction output is
           expected.
    :param int targetcol: Index of batch column that contain targets.
    :return: Result(s) of evaluation, e.g. accuracy, precision, ...
    :rtype: float or tuple of floats if there is more than one metric
    """
    targets = []

    def accumulate(batch):
        p_batch, target = batch[:targetcol], batch[targetcol]
        targets.extend(target)
        return p_batch

    def compute_metric(metric, targets, preds):
        result = metric(targets, preds)
        # call eval() on result if Theano function and convert ndarray to float
        return float(result.eval() if hasattr(result, 'eval') else result)

    @nut_function
    def Extract(x, col):
        return x if col is None else x[col]

    preds = (batches >> Map(accumulate) >> network.predict(flatten=False) >>
             Extract(predcol) >> Flatten() >> Collect())
    targets, preds = np.vstack(targets), np.vstack(preds)
    targets = targets.astype(np.float)
    results = tuple(compute_metric(m, targets, preds) for m in metrics)
    return results if len(results) > 1 else results[0]


class Network(object):
    """
    Abstract base class for networks. Allows to wrap existing network APIs
    such as Lasagne or Keras into an API that enables direct usage of the
    network as a Nut in a nuts flow.
    """

    def __init__(self, filepath):
        """
        Constructs base wrapper for networks.

        :param string filepath: Filepath where network weights are saved to
            and loaded from.
        """
        self.filepath = filepath
        self.best_score = None  # score of best scoring network so far

    def train(self):
        """
        Train network

        train_losses = samples >> batcher >> network.train() >> Collect()

        :return: Typically returns training loss per batch.
        """
        raise NotImplementedError('Implement train()!')

    def validate(self):
        """
        Validate network

        val_losses = samples >> batcher >> network.validate() >> Collect()

        :return: Typically returns validation loss per batch.
        """
        raise NotImplementedError('Implement validate()!')

    def predict(self, flatten=True):
        """
        Get network predictions

        predictions = samples >> batcher >> network.predict() >> Collect()

        :param bool flatten: True: return individual predictions instead
          of batch of prediction
        :return: Typically returns softmax class probabilities.
        :rtype: ndarray
        """
        raise NotImplementedError('Implement predict()!')

    def evaluate(self, metrics, predcol=None, targetcol=-1):
        """
        Evaluate performance of network for given metrices

        acc, f1 = samples >> batcher >> network.evaluate([accuracy, f1_score])

        :param list metric: List of metrics. See EvalNut for details.
        :param int|None predcol: Index of column in prediction to extract
                for evaluation. If None a single prediction output is
                expected.
        :param int targetcol: Index of batch column that contain targets.
        :return: Result for each metric as a tuple or a single float if
           there is only one metric.
        """
        return EvalNut(self, metrics, predcol, targetcol)

    def save_best(self, score, minimum=True):
        """
        Save weights of best network

        :param float score: Score of the network, e.g. loss, accuracy
        :param bool minimum: True means lower score is better, e.g. loss
          and the network with the lower score score is saved.
        """

        if (not self.best_score or
                (minimum is True and score < self.best_score) or
                (minimum is False and score > self.best_score)):
            self.best_score = score
            self.save_weights()

    def save_weights(self):
        """
        Save network weights. network.filepath is used.

        network.save_weights()
        """
        raise NotImplementedError('Implement save_weights()!')

    def load_weights(self):
        """
        Load network weights. network.filepath is used.

        network.load_weights()
        """
        raise NotImplementedError('Implement load_weights()!')

    def print_layers(self):
        """Print description of the network layers"""
        raise NotImplementedError('Implement print_layers()!')


class LasagneNetwork(Network):  # pragma no cover
    """
    Wrapper for Lasagne models: https://lasagne.readthedocs.io/en/latest/
    """

    def __init__(self, out_layer, train_fn, val_fn, pred_fn,
                 filepath='weights_lasagne_net.npz'):
        """
        Construct wrapper around Lasagne network.

        :param Lasgane layer out_layer: Output layer of Lasagne network.
        :param Theano function train_fn: Training function
        :param Theano function val_fn: Validation function
        :param Theano function pred_fn: Prediction function
        :param string filepath: Filepath to save/load model weights.
        """
        Network.__init__(self, filepath)
        self.out_layer = out_layer
        self.train_fn = train_fn
        self.val_fn = val_fn
        self.pred_fn = pred_fn

    @staticmethod
    def _weight_layers(layer):
        """Return list of layers with weights. InputLayer is NOT returned."""
        while hasattr(layer, 'input_layer'):
            yield layer
            layer = layer.input_layer

    @staticmethod
    def _get_named_params(network):
        """Return layer parameters and names"""
        for l_num, layer in enumerate(LasagneNetwork._weight_layers(network)):
            for p_num, param in enumerate(layer.get_params()):
                name = '{}_{}'.format(l_num, p_num)
                yield name, param

    def train(self):
        return TrainValNut(self.train_fn)

    def validate(self):
        return TrainValNut(self.val_fn)

    def predict(self, flatten=True):
        return PredictNut(self.pred_fn, flatten)

    def save_weights(self):
        weights = {name: p.get_value() for name, p in
                   LasagneNetwork._get_named_params(self.out_layer)}
        np.savez_compressed(self.filepath, **weights)

    def load_weights(self):
        weights = np.load(self.filepath)
        for name, param in LasagneNetwork._get_named_params(self.out_layer):
            param.set_value(weights[name])

    def print_layers(self):
        for layer in LasagneNetwork._weight_layers(self.out_layer):
            print '_' * 80
            print layer.__class__.__name__


class KerasNetwork(Network):  # pragma no cover
    """
    Wrapper for Keras models: https://keras.io/
    """

    def __init__(self, model, filepath='weights_keras_net.hd5'):
        """
        Construct wrapper around Keras model.

        :param Keras model model: Keras model to wrap. See
            https://keras.io/models/sequential/
            https://keras.io/models/model/

        :param string filepath: Filepath to save/load model weights.
        """
        Network.__init__(self, filepath)
        self.model = model

    def train(self):
        return TrainValNut(self.model.train_on_batch)

    def validate(self):
        return TrainValNut(self.model.test_on_batch)

    def predict(self, flatten=True):
        return PredictNut(self.model.predict_on_batch, flatten)

    def save_weights(self):
        self.model.save_weights(self.filepath)

    def load_weights(self):
        self.model.load_weights(self.filepath)

    def print_layers(self):
        self.model.summary()
