"""
.. module:: network
   :synopsis: Wrapper around other network APIs such as Lasagne, Keras and
              Pytorch to enable usage within nuts-flow/ml.
              For instance, with a wrapped network one can write:
              samples >> build_batch >> network.train() >> log_loss >> Consume()
"""
from __future__ import print_function

import numpy as np

from nutsflow.common import itemize
from nutsflow import (nut_processor, nut_sink, Collect, Map,
                      Flatten, Get)


@nut_processor
def TrainValNut(batches, func, **kwargs):
    """
    batches >> TrainValNut(func, **kwargs)

    Create nut to train or validate a network.

    :param iterable over batches batches: Batches to train/validate.
    :param function func: Training or validation function of network.
    :param kwargs kwargs: Keyword arguments passed on to function.
    :return: Result(s) of training/validation function, e.g. loss, accuracy, ...
    :rtype: float or array/tuple of floats
    """
    for batch in batches:
        yield func(*batch, **kwargs)


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
        pred_batch = func(batch)
        if flatten:
            for prediction in pred_batch:
                yield prediction
        else:
            yield pred_batch


@nut_sink
def EvalNut(batches, network, metrics, compute, predcol=None):
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
    :param function compute: Function of the form f(metric, targets, preds)
           that computes the given metric (e.g. mean accuracy) for the given
           targets and predictions.
    :param int|None predcol: Index of column in prediction to extract
           for evaluation. If None a single prediction output is
           expected.
    :return: Result(s) of evaluation, e.g. accuracy, precision, ...
    :rtype: float or tuple of floats if there is more than one metric
    """
    targets = []

    def accumulate(batch):
        inputs, outputs = batch
        target = outputs[0] if isinstance(outputs, list) else outputs
        targets.extend(target)
        return inputs

    preds = (batches >> Map(accumulate) >> network.predict(flatten=False) >>
             Get(predcol) >> Flatten() >> Collect())

    targets, preds = np.vstack(targets), np.vstack(preds)
    targets = targets.astype(np.float)
    results = tuple(compute(m, targets, preds) for m in metrics)
    return results if len(results) > 1 else results[0]


class Network(object):
    """
    Abstract base class for networks. Allows to wrap existing network APIs
    such as Lasagne, Keras or Pytorch into an API that enables direct usage
    of the network as a Nut in a nuts flow.
    """

    def __init__(self, weightspath):
        """
        Constructs base wrapper for networks.

        :param string weightspath: Filepath where network weights are saved to
            and loaded from.
        """
        self.weightspath = weightspath
        self.best_score = None  # score of best scoring network so far

    def _weightspath(self, weightspath):
        """
        Return give weightspath if not None else return self.weightspath.
        
        :param string|None weightspath: Path to network weights or None.
        :return: Return weightspath
        """
        return self.weightspath if weightspath is None else weightspath

    def train(self):
        """
        Train network

        >>> train_losses = samples >> batcher >> network.train() >> Collect() # doctest: +SKIP

        :return: Typically returns training loss per batch.
        """
        raise NotImplementedError('Implement train()!')

    def validate(self):
        """
        Validate network

        >>> val_losses = samples >> batcher >> network.validate() >> Collect()  # doctest: +SKIP

        :return: Typically returns validation loss per batch.
        """
        raise NotImplementedError('Implement validate()!')

    def predict(self, flatten=True):
        """
        Get network predictions

        >>> predictions = samples >> batcher >> network.predict() >> Collect()  # doctest: +SKIP

        :param bool flatten: True: return individual predictions instead
          of batched prediction
        :return: Typically returns softmax class probabilities.
        :rtype: ndarray
        """
        raise NotImplementedError('Implement predict()!')

    def evaluate(self, metrics, predcol=None, targetcol=-1):
        """
        Evaluate performance of network for given metrices

        >>> acc, f1 = samples >> batcher >> network.evaluate([accuracy, f1_score])  # doctest: +SKIP

        :param list metric: List of metrics. See EvalNut for details.
        :param int|None predcol: Index of column in prediction to extract
                for evaluation. If None a single prediction output is
                expected.
        :param int targetcol: Index of batch column that contain targets.
        :return: Result for each metric as a tuple or a single float if
           there is only one metric.
        """
        raise NotImplementedError('Implement evaluate()!')

    def save_best(self, score, isloss=True):
        """
        Save weights of best network

        :param float score: Score of the network, e.g. loss, accuracy
        :param bool isloss: True means lower score is better, e.g. loss
          and the network with the lower score score is saved.
        """

        if (not self.best_score or
                (isloss is True and score <= self.best_score) or
                (isloss is False and score >= self.best_score)):
            self.best_score = score
            self.save_weights()

    def save_weights(self, weightspath=None):
        """
        Save network weights.

        | network.save_weights()

        :param string weightspath: Path to network weights.
          self.weightspath is used if weightspath is None.
        """
        raise NotImplementedError('Implement save_weights()!')

    def load_weights(self, weightspath=None):
        """
        Load network weights.

        | network.load_weights()
        
        :param string weightspath: Path to network weights.
          self.weightspath is used if weightspath is None.
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
                 weightspath='weights_lasagne_net.npz'):
        """
        Construct wrapper around Lasagne network.

        :param Lasgane layer out_layer: Output layer of Lasagne network.
        :param Theano function train_fn: Training function
        :param Theano function val_fn: Validation function
        :param Theano function pred_fn: Prediction function
        :param string weightspath: Filepath to save/load model weights.
        """
        Network.__init__(self, weightspath)
        self.out_layer = out_layer
        self.train_fn = train_fn
        self.val_fn = val_fn
        self.pred_fn = pred_fn

    @staticmethod
    def _layers(layer, ret_input=False):
        """Return network layers. InputLayer is returned if ret_input==True."""
        while hasattr(layer, 'input_layer'):
            yield layer
            layer = layer.input_layer
        if ret_input:
            yield layer

    @staticmethod
    def _get_named_params(network):
        """Return layer parameters and names"""
        for l_num, layer in enumerate(LasagneNetwork._layers(network)):
            for p_num, param in enumerate(layer.get_params()):
                name = '{}_{}'.format(l_num, p_num)
                yield name, param

    def train(self, **kwargs):
        return TrainValNut(self.train_fn, **kwargs)

    def validate(self, **kwargs):
        return TrainValNut(self.val_fn, **kwargs)

    def predict(self, flatten=True):
        return PredictNut(self.pred_fn, flatten)

    def evaluate(self, metrics, predcol=None):
        def compute(metric, targets, preds):
            result = metric(targets, preds)
            return result.eval() if hasattr(result, 'eval') else result

        return EvalNut(self, metrics, compute, predcol)

    def save_weights(self, weightspath=None):
        weightspath = super(LasagneNetwork, self)._weightspath(weightspath)
        weights = {name: p.get_value() for name, p in
                   LasagneNetwork._get_named_params(self.out_layer)}
        np.savez_compressed(weightspath, **weights)

    def load_weights(self, weightspath=None):
        weightspath = super(LasagneNetwork, self)._weightspath(weightspath)
        weights = np.load(weightspath)
        for name, param in LasagneNetwork._get_named_params(self.out_layer):
            param.set_value(weights[name])

    def print_layers(self):
        import lasagne as la
        layers = list(LasagneNetwork._layers(self.out_layer, ret_input=True))
        for i, layer in enumerate(reversed(layers)):
            name = layer.__class__.__name__
            shape = la.layers.get_output_shape(layer)
            print('{:3d}  {:30s} {}'.format(i, name, shape), end=' ')
            if hasattr(layer, 'filter_size'):
                print('{}'.format(layer.filter_size[0]), end='//')
            elif hasattr(layer, 'pool_size'):
                is_int = isinstance(layer.pool_size, int)
                size = layer.pool_size if is_int else layer.pool_size[0]
                print('{}'.format(size), end='//')
            if hasattr(layer, 'p'):
                print(' [{:.2f}]'.format(layer.p), end='')
            if hasattr(layer, 'stride'):
                print('{}'.format(layer.stride[0]), end='')
            if hasattr(layer, 'learning_rate_scale'):
                if layer.learning_rate_scale != 1.0:
                    lr_scale = layer.learning_rate_scale
                    print(' [lr_scale={:.2f}]'.format(lr_scale), end='')
            if hasattr(layer, 'params'):
                for param in layer.params:
                    if 'trainable' not in layer.params[param]:
                        print(' [NT]', end='')
            print()


class KerasNetwork(Network):  # pragma no cover
    """
    Wrapper for Keras models: https://keras.io/
    """

    def __init__(self, model, weightspath='weights_keras_net.hd5'):
        """
        Construct wrapper around Keras model.

        :param Keras model model: Keras model to wrap. See
            https://keras.io/models/sequential/
            https://keras.io/models/model/

        :param string weightspath: Filepath to save/load model weights.
        """
        Network.__init__(self, weightspath)
        self.model = model

    # Since Keras with tensorflow 2.x the function train_on_batch()
    # does not accept a batch format of [[inputs],[outputs]] anymore,
    # while other similar function such as test_on_batch, predict_on_batch
    # are still fine with it. Therefore only fixing for train_on_batch
    # where sublist are removed if inputs and/or outputs are single items.
    def _train_on_batch(self, x_batches, y_batches, **kwargs):
        x_batches, y_batches = itemize(x_batches), itemize(y_batches)
        return self.model.train_on_batch(x_batches, y_batches, kwargs)

    def train(self, **kwargs):
        return TrainValNut(self._train_on_batch, **kwargs)

    def validate(self, **kwargs):
        return TrainValNut(self.model.test_on_batch, **kwargs)

    def predict(self, flatten=True):
        return PredictNut(self.model.predict_on_batch, flatten)

    def evaluate(self, metrics, predcol=None):
        def compute(metric, targets, preds):
            result = metric(targets, preds).numpy()
            is_vector = hasattr(result, '__iter__')
            return float(np.mean(result) if is_vector else result)

        return EvalNut(self, metrics, compute, predcol)

    def save_weights(self, weightspath=None):
        weightspath = super(KerasNetwork, self)._weightspath(weightspath)
        self.model.save_weights(weightspath)

    def load_weights(self, weightspath=None):
        weightspath = super(KerasNetwork, self)._weightspath(weightspath)
        self.model.load_weights(weightspath)

    def print_layers(self):
        self.model.summary()


class PytorchNetwork(Network):  # pragma no cover
    """
    Wrapper for Pytorch models:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
    """

    def __init__(self, model, weightspath='weights_pytorch_net.pt'):
        """
        Construct wrapper around Pytorch model.

        :param Pytorch model model: Pytorch model to wrap.
               model needs to have three attributes:
               | model.device:, e.g 'cuda:0' or 'cpu'
               | model.optimizer: e.g. torch.optim.SGD
               | model.losses: (list of) loss functions, e.g. F.cross_entropy
        :param string weightspath: Filepath to save/load model weights.
        """
        Network.__init__(self, weightspath)

        assert hasattr(model, 'device')
        assert hasattr(model, 'optimizer')
        assert hasattr(model, 'losses')
        self.model = model
        model.to(model.device)

    def _to_tensor(self, batches, flatten):
        """
        Convert batches into Pytorch tensors.

        :param list|ndarray batches: Numpy array or list of arrays.
        :param bool flatten: If true and batch contains only one column
               return single tensor instead of list of tensors.
        :return: List of batches as PyTorch tensors or a single tensor
        :rtype: [tensors] or tensor
        """
        import torch
        T = lambda b: torch.as_tensor(b, device=self.model.device)
        batches = self._to_list(batches)
        tensors = [T(b) for b in batches if not isinstance(b, str)]
        if flatten and len(tensors) == 1:
            return tensors[0]
        return tensors

    def _to_list(self, x):
        """
        Wraps x in a list if it is not already a list.

        :param object x: Any object.
        :return: x wrapped in list
        :rtype: list
        """
        return x if isinstance(x, list) else [x]

    def _train_batch(self, x_batches, y_batches, *args):
        """
        Performs a single gradient step on a batch.

        :param ndarray|[ndarray] x_batches: Input batch or list of batches
        :param ndarray|[ndarray] y_batches: Output batch or list of batches
        :return: losses. If there is multiple outputs then a list with
                the losses for each output and the mean over these losses
                is returned. Otherwise a single float with the loss is returned.
        :rtype: float|[float]
        """
        x_tensors = self._to_tensor(x_batches, True)
        y_tensors = self._to_tensor(y_batches, False)
        model = self.model
        model.optimizer.zero_grad()
        y_preds = self._to_list(model(x_tensors, *args))
        loss_fns = self._to_list(model.losses)
        losses = []
        for loss_fn, y_pred, y_true in zip(loss_fns, y_preds, y_tensors):
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            losses.append(loss.item())
        model.optimizer.step()
        return [np.mean(losses)] + losses if len(losses) > 1 else losses[0]

    def _validate_batch(self, x_batches, y_batches, *args):
        """
        Performs a forward step to compute losses.

        :param [ndarray] x_batches: List of input batches
        :param [ndarray] y_batches: List of output/target batches
        :return: losses. If there is multiple outputs then a list with
                the losses for each output and the mean over these losses
                is returned. Otherwise a single float with the loss is returned.
        :rtype: float|[float]
        """
        import torch
        losses = []
        with torch.no_grad():
            x_tensors = self._to_tensor(x_batches, True)
            y_tensors = self._to_tensor(y_batches, False)
            model = self.model
            y_preds = self._to_list(model(x_tensors, *args))
            loss_fns = self._to_list(model.losses)
            for loss_fn, y_pred, y_true in zip(loss_fns, y_preds, y_tensors):
                loss = loss_fn(y_pred, y_true)
                losses.append(loss.item())
        return [np.mean(losses)] + losses if len(losses) > 1 else losses[0]

    def _predict_batch(self, x_batches, *args):
        """
        Performs a forward step to compute output.

        :param [ndarray] x_batches: List of input batches
        :return: network outputs
        :rtype: list
        """
        import torch
        with torch.no_grad():
            x_tensors = self._to_tensor(x_batches, True)
            y_preds = self.model(x_tensors, *args)
            return [p.cpu().numpy() for p in y_preds]

    def train(self, **kwargs):
        self.model.train()
        return TrainValNut(self._train_batch, **kwargs)

    def validate(self, **kwargs):
        self.model.eval()
        return TrainValNut(self._validate_batch, **kwargs)

    def predict(self, flatten=True):
        self.model.eval()
        return PredictNut(self._predict_batch, flatten)

    def evaluate(self, metrics, predcol=None):
        def compute(metric, targets, preds):
            result = metric(targets, preds)
            return result.item() if hasattr(result, 'item') else result

        self.model.eval()
        return EvalNut(self, metrics, compute, predcol)

    def save_weights(self, weightspath=None):
        import torch
        weightspath = super(PytorchNetwork, self)._weightspath(weightspath)
        torch.save(self.model.state_dict(), weightspath)

    def load_weights(self, weightspath=None):
        import torch
        weightspath = super(PytorchNetwork, self)._weightspath(weightspath)
        self.model.load_state_dict(torch.load(weightspath))

    def print_layers(self, input_shape=None):
        """
        Print network architecture (and layer dimensions).

        :param tuple|None input_shape: (C, H, W) or None
               If None, layer dimensions and param numbers are not printed.
        """
        if input_shape:
            from torchsummary import summary
            device = self.model.device[:4]  # remove GPU id, e.g. cuda:0
            summary(self.model, input_shape, device=device)
        else:
            print(str(self.model))
