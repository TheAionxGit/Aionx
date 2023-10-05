import tensorflow as tf
from tensorflow import keras


class DNN(keras.models.Model):
    """
    ----------------------------------------------------------------------------------------
    A regular Dense neural network built using TensorFlow Subclassing API.
    ----------------------------------------------------------------------------------------

    PARAMETERS:
    ----------
    n_layers : int
        Number of hidden layers in the network.
        
    n_neurons : int
        Number of neurons for each layer in the network.
        
    activation : str
        The activation function used for each layer in the network.
        
    n_output : int, optional (default=1)
        The number of output neurons.
        
    output_activation : int, optional(default='linear')
        
    dropout : float, optional (default=0.2)
        The dropout rate applied to each layer in the network, except for the input layer.
        
    weights_initialization_std : float, optional (default=0.03)
        The initializer for the standard deviation of the weights of the layers.
        
    weights_regularizer : float, optional (default=0.00)
        The L1 & L2 penalty to add to the gradient during training.

    METHODS:
    -------
    call(x, training=False):
        The forward pass of the network. If training is True, then dropout will be applied.
    ----------------------------------------------------------------------------------------
    """
    def __init__(self, n_layers:int, n_neurons:int, activation:str, n_outputs:int=1,
                 output_activation:str="linear",
                 weights_initialization_std:float=0.03, dropout:float=0.2,
                 weights_regularizer:float=0.00,
                 **kwargs):
        
        super(DNN, self).__init__(**kwargs)

        # Main Attributes.
        self.n_layers   = n_layers
        self.n_neurons  = n_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.n_outputs  = n_outputs
        self.dropout = dropout
        self.weights_initialization_std = weights_initialization_std
        self.weights_regularizer = weights_regularizer

        # Hidden layers of the network..
        self.hidden_layers = [
            keras.layers.Dense(
                units=self.n_neurons,
                activation=self.activation,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                            l2=self.weights_regularizer)
            ) for _ in range(self.n_layers)
        ]

        # Dropout layers applied at each hidden layer.
        self.dropout_layers = [
            keras.layers.Dropout(self.dropout)
            for _ in range(self.n_layers)
        ]

        # Output layer.
        self.output_layer = keras.layers.Dense(self.n_outputs, activation=self.output_activation)

    def call(self, x, training=False):
        """
        -----------------------------------------------------------------------------------
        The forward pass of the network. If training is True, then dropout will be applied.
        -----------------------------------------------------------------------------------
        
        PARAMETERS:
        ----------
        x : Tensor
            The input tensor to the network.
        training : bool, optional (default=False)
            A flag indicating whether the network is in training mode or not.

        RETURNS:
        -------
        output : Tensor
            The output of the network.
        """
        # Flatten the input.
        x = tf.keras.layers.Flatten()(x)

        # Apply the hidden layers and dropout.
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.dropout_layers[i](x, training)

        # Apply the output layer.
        output = self.output_layer(x)

        return output


class RNN(keras.models.Model):
    """
    ----------------------------------------------------------------------------------------
    A recurrent neural network built using TensorFlow Subclassing API.
    ----------------------------------------------------------------------------------------

    PARAMETERS:
    ----------
    n_layers : int
        Number of hidden layers in the network.
        
    n_neurons : int
        Number of neurons for each layer in the network.
        
    recurrent_layers : str
        One of {'SimpleRNN', 'LSTM', 'GRU'}. 
        
    activation : str
        The activation function used for each layer in the network.
        
    n_output : int, optional (default=1)
        The number of output neurons.
        
    output_activation : int, optional(default='linear')
        The activation function to be applied on the output of the network.
        
    dropout : float, optional (default=0.2)
        The dropout rate applied to each layer in the network, except for the input layer.
        
    weights_initialization_std : float, optional (default=0.03)
        The initializer for the standard deviation of the weights of the layers.
        
    weights_regularizer : float, optional (default=0.00)
        The L1 & L2 penalty to add to the gradient during training.
        
    return_sequences : bool, optional (default=False)
        if True, the last hidden layer will return the whole sequence and the output layers will be
        wrapped in a TimeDistributed layer. If False, the network will only return the last time steps.
        See keras implementation for more information.
        
    METHODS:
    -------
    call(x, training=False):
        The forward pass of the network. If training is True, then dropout will be applied.
    ----------------------------------------------------------------------------------------

    """
    
    def __init__(self, n_layers:int, n_neurons:int, recurrent_layers:str="SimpleRNN",
                 activation:str="tanh", n_outputs:int=1, return_sequences:bool=False,
                 output_activation:str="linear",
                 weights_initialization_std:float=0.03, dropout:float=0.2,
                 weights_regularizer:float=0.00,
                 **kwargs):
        
        
        super(RNN, self).__init__(**kwargs)

        # Main Attributes.
        self.n_layers   = n_layers
        self.n_neurons  = n_neurons
        self.recurrent_layers = recurrent_layers
        self.activation = activation
        self.output_activation = output_activation
        self.n_outputs  = n_outputs
        self.dropout = dropout
        self.weights_initialization_std = weights_initialization_std
        self.weights_regularizer = weights_regularizer
        self.return_sequences=return_sequences
        
        if self.recurrent_layers == "SimpleRNN":
            self.hidden_layers = [
                keras.layers.SimpleRNN(
                    units=self.n_neurons,
                    activation=self.activation,
                    dropout=self.dropout,
                    return_sequences=True,
                    kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                                l2=self.weights_regularizer)
                ) for _ in range(self.n_layers-1)
            ]    
            self.hidden_layers.append(keras.layers.SimpleRNN(
                    units=self.n_neurons,
                    activation=self.activation,
                    dropout=self.dropout,
                    return_sequences=self.return_sequences,
                    kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                                l2=self.weights_regularizer)
                ))


        elif self.recurrent_layers == "LSTM":
            self.hidden_layers = [
                keras.layers.LSTM(
                    units=self.n_neurons,
                    activation=self.activation,
                    dropout=self.dropout,
                    return_sequences=True,
                    kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                                l2=self.weights_regularizer)
                ) for _ in range(self.n_layers-1)
            ]
            self.hidden_layers.append(keras.layers.LSTM(
                    units=self.n_neurons,
                    activation=self.activation,
                    dropout=self.dropout,
                    return_sequences=self.return_sequences,
                    kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                                l2=self.weights_regularizer)
                ))
            
        elif self.recurrent_layers == "GRU":
            self.hidden_layers = [
                keras.layers.GRU(
                    units=self.n_neurons,
                    activation=self.activation,
                    dropout=self.dropout,
                    return_sequences=True,
                    kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                                l2=self.weights_regularizer)
                ) for _ in range(self.n_layers-1)
            ]
            self.hidden_layers.append(keras.layers.GRU(
                    units=self.n_neurons,
                    activation=self.activation,
                    dropout=self.dropout,
                    return_sequences=self.return_sequences,
                    kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.weights_initialization_std),
                    kernel_regularizer=keras.regularizers.l1_l2(l1=self.weights_regularizer,
                                                                l2=self.weights_regularizer)
                ))
        else:
            raise ValueError(
                f"Recurrent_layers must be one of {'SimpleRNN', 'LSTM', 'GRU'}. Input received: {self.recurrent_layers}")
            
        # Output layer
        if self.return_sequences:
            self.output_layer = keras.layers.TimeDistributed(
                keras.layers.Dense(self.n_outputs, activation=self.output_activation)
            )
        else:
            self.output_layer = keras.layers.Dense(self.n_outputs, activation=self.output_activation) 
        
        
    def call(self, x, training=False):
        """
        -----------------------------------------------------------------------------------
        The forward pass of the network. If training is True, then dropout will be applied.
        -----------------------------------------------------------------------------------
        
        PARAMETERS:
        ----------
        x : Tensor
            The input tensor to the network.
        training : bool, optional (default=False)
            A flag indicating whether the network is in training mode or not.

        RETURNS:
        -------
        output : Tensor
            The output of the network.
        """

        # Loop through all the hidden layers and apply them to the input tensor
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, training=training)

        # Pass the output tensor through the output layer
        output = self.output_layer(x)

        return output

