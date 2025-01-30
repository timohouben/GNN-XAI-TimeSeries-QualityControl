import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import spektral

tf.keras.utils.get_custom_objects().clear()
@tf.function
def timeseries_pooling(inputs, aggregation_type='mean', batch_size=128, timestep_before=120,
                       timestep_after=60, features=32, type='pool', freq=1):
    sensor_features, sensor_indices, sensor_ind = inputs
    sequence_length = int((timestep_before + timestep_after)/freq + 1)
    features_shape = tf.shape(sensor_features)[-1]
    if type == 'pool':
        aggregation_func = {
            "sum": tf.math.reduce_sum,
            "mean": tf.math.reduce_mean,
            "max": tf.math.reduce_max,
        }.get(aggregation_type)
        sensor_features_partitioned = tf.dynamic_partition(
                 sensor_features, sensor_indices, batch_size)
        avg_features_list = tf.TensorArray(tf.float32, size=batch_size)
        for i in range(batch_size):
            sensor_features_time = tf.reshape(sensor_features_partitioned[i], [sequence_length, -1, features])
            sensor_features_time = tf.transpose(sensor_features_time, [1, 0, 2])
            avg_features = aggregation_func(sensor_features_time, axis=0, keepdims=False)
            avg_features_list = avg_features_list.write(i, avg_features)   
        avg_features_list = avg_features_list.stack()
    
        if aggregation_type == 'max':
            gather_indices = tf.where(tf.math.is_finite(tf.reduce_sum(avg_features_list, (1, 2))) == True)
        else:
            gather_indices = tf.where((tf.reduce_sum(avg_features_list, (1, 2)) != 0) &
                                     (tf.math.is_finite(tf.reduce_sum(avg_features_list, (1, 2))) == True))
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        avg_features_list = tf.gather(avg_features_list, gather_indices, axis=0)
    elif type == 'selection':
        avg_features_list = tf.gather(sensor_features, sensor_ind)
        avg_features_list = tf.reshape(avg_features_list, [sequence_length, -1, features])
        avg_features_list = tf.transpose(avg_features_list, [1, 0, 2])
    return avg_features_list

@tf.keras.utils.register_keras_serializable()      
class TimeLayer(layers.Layer):
    def __init__(self, filter_1_size=8, n_stacks=2, layer_type='lstm', activation='tanh', kernel_size=5,
                 regularizer=None, pool_size=3, alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        self.layer_type = layer_type
        self.activation = activation
        self.kernel_size = kernel_size
        self.regularizer = regularizer
        self.time_layers = []
        self.pooling_layers = []
        self.filter_1_size = filter_1_size
        self.n_stacks = n_stacks
        self.alpha = alpha
        self.pool_size = pool_size
        if regularizer is not None:
            regularizer = l2(regularizer)
        
        if layer_type == 'lstm':
            self.time1 = layers.LSTM(filter_1_size, activation=activation, recurrent_dropout=0,
                                                        kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                                        return_sequences=True)
            self.time2 = layers.LSTM(filter_1_size, activation=activation, recurrent_dropout=0,
                                                        kernel_regularizer=regularizer,
                                     recurrent_regularizer=regularizer, return_sequences=True)
            self.max_pooling = layers.MaxPooling1D(pool_size=pool_size)
            for i in range(n_stacks):
                self.time_layers.append(layers.LSTM(filter_1_size*(2**(i+1)), activation=activation, recurrent_dropout=0,
                                                    kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                                    return_sequences=True))
                self.time_layers.append(layers.LSTM(filter_1_size*(2**(i+1)), activation=activation, recurrent_dropout=0,
                                                    kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                                    return_sequences=True))
                self.pooling_layers.append(layers.MaxPooling1D(pool_size=pool_size))
                
            self.time4 = layers.LSTM(filter_1_size*(2**(n_stacks+1)), activation=activation, recurrent_dropout=0,
                                     kernel_regularizer=regularizer, recurrent_regularizer=regularizer)
        else:
            # Convolutional part
            self.time1 = layers.Conv1D(filters=filter_1_size, kernel_size=kernel_size, kernel_regularizer=regularizer, padding='same')
            self.time2 = layers.Conv1D(filters=filter_1_size, kernel_size=kernel_size, kernel_regularizer=regularizer, padding='same')
            self.leakyrelu1 = layers.LeakyReLU(alpha=alpha)
            self.leakyrelu2 = layers.LeakyReLU(alpha=alpha)
            self.max_pooling = layers.MaxPooling1D(pool_size=pool_size)
    
            self.leakyrelu_layers = []
            for i in range(n_stacks):
                self.time_layers.append(layers.Conv1D(filters=filter_1_size*(2**(i+1)), kernel_regularizer=regularizer,
                                                      kernel_size=kernel_size, padding='same'))
                self.leakyrelu_layers.append(layers.LeakyReLU(alpha=alpha))
                self.time_layers.append(layers.Conv1D(filters=filter_1_size*(2**(i+1)), kernel_regularizer=regularizer,
                                                      kernel_size=kernel_size, padding='same'))
                self.leakyrelu_layers.append(layers.LeakyReLU(alpha=alpha))
                self.pooling_layers.append(layers.MaxPooling1D(pool_size=3))
    
            self.time4 = layers.Conv1D(filters=filter_1_size*(2**(n_stacks+1)), kernel_regularizer=regularizer,
                                       kernel_size=kernel_size, padding='same')
            self.leakyrelu3 = layers.LeakyReLU(alpha=alpha)
            self.global_pooling = layers.GlobalAveragePooling1D()   
   
    def call(self, inputs):
        x1 = self.time1(inputs)
        if self.layer_type == 'cnn':
            x1 = self.leakyrelu1(x1)
        x1 = self.time2(x1)
        if self.layer_type == 'cnn':
            x1 = self.leakyrelu2(x1)
        x1 = self.max_pooling(x1)  
        for i in range(len(self.pooling_layers)):
            x1 = self.time_layers[2*i](x1)
            if self.layer_type == 'cnn':
                x1 = self.leakyrelu_layers[2*i](x1)
            x1 = self.time_layers[2*i+1](x1)
            if self.layer_type == 'cnn':
                x1 = self.leakyrelu_layers[2*i+1](x1)
            x1 = self.pooling_layers[i](x1)
        x1 = self.time4(x1)
        if self.layer_type == 'cnn':
            x1 = self.leakyrelu3(x1)
            x1 = self.global_pooling(x1)
        return x1
         
    def get_config(self):
        config = super().get_config()
        config.update(
            {"layer_type": self.layer_type,
             "activation": self.activation,
            "kernel_size": self.kernel_size,
            "regularizer": self.regularizer,
            "filter_1_size": self.filter_1_size,
            "n_stacks": self.n_stacks,
            "alpha": self.alpha,
            "pool_size": self.pool_size})
        return config



class GCNClassifier(tf.keras.Model):
    def __init__(self, model_config, preprocessing_config):
        super().__init__()
        self.model_config = model_config
        self.timestep_before = preprocessing_config.timestep_before 
        self.timestep_after = preprocessing_config.timestep_after
        self.batch_size = preprocessing_config.batch_size
        self.aggregation_type = model_config.pooling.aggregation_type
        self.ds_type =  preprocessing_config.ds_type
        self.features_gcn_out = self.model_config.graph_convolution.units
        
        if self.ds_type == 'cml':
            self.freq = 1
            input_feature_numb = 2
        elif self.ds_type == 'soilnet':
            self.freq = 15
            input_feature_numb = 3
            self.reshape_units = self.features_gcn_out + input_feature_numb

        self.model_info = tf.Variable([self.timestep_before, self.timestep_after, self.batch_size, self.freq],
                                      trainable=False)
        #self.model_info = tf.Variable([self.timestep_before, self.timestep_after, self.batch_size],
        #                              trainable=False)        
        self.ds_type = preprocessing_config.ds_type
        self.model_type = tf.Variable(self.ds_type, trainable=False)
        self.model_normalization = tf.Variable(preprocessing_config.normalization, trainable=False)
        
        if self.model_config.graph_convolution.regularizer is not None:
            kernel_regularizer = l2(self.model_config.graph_convolution.regularizer)
        else:
            kernel_regularizer = None

        self.features_gcn_out = self.model_config.graph_convolution.units
        if self.model_config.graph_convolution.layer == "AGNNConv":
            self.gcn_layer = spektral.layers.AGNNConv(aggregate=self.model_config.graph_convolution.aggregation_type,
                                                 activation=self.model_config.graph_convolution.activation) 
            self.features_gcn_out = input_feature_numb
        elif self.model_config.graph_convolution.layer == "GATConv":
            self.gcn_layer = spektral.layers.GATConv(self.model_config.graph_convolution.units,
                                               attn_heads=self.model_config.graph_convolution.attention_heads,
                                               dropout_rate=self.model_config.graph_convolution.dropout_rate,
                                               activation=self.model_config.graph_convolution.activation,
                                               kernel_regularizer=kernel_regularizer)
            self.features_gcn_out = self.model_config.graph_convolution.attention_heads*self.model_config.graph_convolution.units
        elif self.model_config.graph_convolution.layer == "GeneralConv":
            self.gcn_layer = spektral.layers.GeneralConv(channels=self.model_config.graph_convolution.units,
                                                       dropout=self.model_config.graph_convolution.dropout_rate,
                                                       aggregate=self.model_config.graph_convolution.aggregation_type,
                                    activation=self.model_config.graph_convolution.activation,
                                                       kernel_regularizer=kernel_regularizer)
        elif self.model_config.graph_convolution.layer == "GatedGraphConv":
            self.gcn_layer = spektral.layers.GatedGraphConv(self.model_config.graph_convolution.units,
                                                         n_layers=self.model_config.graph_convolution.n_layers, # n_layers - number of iterations of the GRU cell
                                                         activation=self.model_config.graph_convolution.activation,
                                                         kernel_regularizer=kernel_regularizer)
            
        self.time_layer = TimeLayer(filter_1_size=self.model_config.sequence_layer.filter_1_size,
                                    n_stacks = self.model_config.sequence_layer.n_stacks,
                                    alpha = self.model_config.sequence_layer.alpha,
                                    layer_type=self.model_config.sequence_layer.algorithm,
                                    activation=self.model_config.sequence_layer.activation,
                                    kernel_size=self.model_config.sequence_layer.kernel_size,
                                    regularizer=self.model_config.sequence_layer.regularizer,
                                    pool_size=self.model_config.sequence_layer.pool_size)

        if self.model_config.dense.regularizer is not None:
            dense_regularizer = l2(self.model_config.dense.regularizer)
        else:
            dense_regularizer = None
        self.dense = layers.Dense(self.model_config.dense.units, kernel_regularizer=dense_regularizer)
        self.leakyrelu4 = layers.LeakyReLU(alpha=self.model_config.dense.alpha)
        self.dense2 = layers.Dense(self.model_config.dense.units, kernel_regularizer=dense_regularizer)
        self.leakyrelu5 = layers.LeakyReLU(alpha=self.model_config.dense.alpha)
        self.dense_out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        if self.ds_type == 'cml':
            features, anomalous_sensor_features, adjacency_matrix, sample_indices, sensor_ind = inputs
            # Perform selected graph convolution
            x = self.gcn_layer([features, adjacency_matrix])
            x = timeseries_pooling([x, sample_indices, sensor_ind], self.aggregation_type, self.batch_size,
                                      self.timestep_before, self.timestep_after, self.features_gcn_out, freq=self.freq)
            # Concatetane anomalous time series with pooling results
            x = layers.Concatenate()([anomalous_sensor_features, x])
        elif self.ds_type == 'soilnet':
            features, adjacency_matrix, sample_indices = inputs
            # Perform selected graph convolution
            x = self.gcn_layer([features, adjacency_matrix])
            # concatenate with the input features
            x = layers.Concatenate()([x, features])
            x = graph_reshape([x, sample_indices], self.batch_size, self.timestep_before,
                        self.timestep_after, self.reshape_units) 
            
        timeseries_out = self.time_layer(x)
        # Add dense layer
        dense = self.dense(timeseries_out)
        dense = self.leakyrelu4(dense)
        dense = self.dense2(dense)
        dense = self.leakyrelu5(dense)
        out = self.dense_out(dense)
        return out

def graph_reshape(inputs, batch_size=128, timestep_before=4320,
                       timestep_after=720, features=32, freq=15):
    sensor_features, sensor_indices = inputs
    sequence_length = int((timestep_before + timestep_after)/freq + 1)
    features_shape = tf.shape(sensor_features)[-1]
    features_partitioned = tf.dynamic_partition(sensor_features, sensor_indices, batch_size)
    features_list1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True) #, infer_shape=False
    for i in range(batch_size):
        sensor_features_time = tf.reshape(features_partitioned[i], [sequence_length, -1, features])
        sensor_features_time = tf.transpose(sensor_features_time, [1, 0, 2])
        features_list1 = features_list1.write(i, sensor_features_time)
    features_list = features_list1.concat()
    gather_indices = tf.where((tf.reduce_sum(features_list, (1, 2)) != 0) &
                                     (tf.math.is_finite(tf.reduce_sum(features_list, (1, 2))) == True))
    gather_indices = tf.squeeze(gather_indices, axis=-1)
    features_list = tf.gather(features_list, gather_indices, axis=0)
    return features_list


class BaselineClassifier(tf.keras.Model):
    def __init__(self, model_config, preprocessing_config):
        super().__init__()
        self.ds_type =  preprocessing_config.ds_type
        if self.ds_type == 'cml':
            self.input_feature_numb = 2
            self.freq = 1
        elif self.ds_type == 'soilnet':
            self.input_feature_numb = 3
            self.freq = 15
            
        self.model_config = model_config
        self.batch_size = preprocessing_config.batch_size
        self.timestep_before = preprocessing_config.timestep_before 
        self.timestep_after = preprocessing_config.timestep_after
        self.normalization = tf.Variable(preprocessing_config.normalization, trainable=False)
        self.model_info = tf.Variable([self.timestep_before, self.timestep_after, self.batch_size, self.freq],
                                      trainable=False)
        filter_1_size = self.model_config.baseline_model.filter_1_size
        activation = self.model_config.baseline_model.activation
        pool_size = self.model_config.baseline_model.pool_size
        n_stacks = self.model_config.baseline_model.n_stacks
        alpha = self.model_config.baseline_model.alpha
        kernel_size = self.model_config.baseline_model.kernel_size
        self.time_layers = []
        self.pooling_layers = []

        if self.model_config.baseline_model.regularizer is not None:
            regularizer = l2(regularizer)
        else:
            regularizer = None
        
        if model_config.baseline_model.type != 'cnn':
            self.time1 = layers.LSTM(filter_1_size, activation=activation, recurrent_dropout=0,
                                                        kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                                        return_sequences=True)
            self.time2 = layers.LSTM(filter_1_size, activation=activation, recurrent_dropout=0,
                                                        kernel_regularizer=regularizer,
                                     recurrent_regularizer=regularizer, return_sequences=True)
            self.max_pooling = layers.MaxPooling1D(pool_size=pool_size)
            for i in range(n_stacks):
                self.time_layers.append(layers.LSTM(filter_1_size*(2**(i+1)), activation=activation, recurrent_dropout=0,
                                                    kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                                    return_sequences=True))
                self.time_layers.append(layers.LSTM(filter_1_size*(2**(i+1)), activation=activation, recurrent_dropout=0,
                                                    kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                                    return_sequences=True))
                self.pooling_layers.append(layers.MaxPooling1D(pool_size=pool_size))
                
            self.time4 = layers.LSTM(filter_1_size*(2**(n_stacks+1)), activation=activation, recurrent_dropout=0,
                                     kernel_regularizer=regularizer, recurrent_regularizer=regularizer)
        else:
            # Convolutional part
            self.time1 = layers.Conv1D(filters=filter_1_size, kernel_size=kernel_size,
                                       kernel_regularizer=regularizer, padding='same')
            self.time2 = layers.Conv1D(filters=filter_1_size, kernel_size=kernel_size,
                                       kernel_regularizer=regularizer, padding='same')
            self.leakyrelu1 = layers.LeakyReLU(alpha=alpha)
            self.leakyrelu2 = layers.LeakyReLU(alpha=alpha)
            self.max_pooling = layers.MaxPooling1D(pool_size=pool_size)
    
            self.leakyrelu_layers = []
            for i in range(n_stacks):
                self.time_layers.append(layers.Conv1D(filters=filter_1_size*(2**(i+1)), kernel_regularizer=regularizer,
                                                      kernel_size=kernel_size, padding='same'))
                self.leakyrelu_layers.append(layers.LeakyReLU(alpha=alpha))
                self.time_layers.append(layers.Conv1D(filters=filter_1_size*(2**(i+1)), kernel_regularizer=regularizer,
                                                      kernel_size=kernel_size, padding='same'))
                self.leakyrelu_layers.append(layers.LeakyReLU(alpha=alpha))
                self.pooling_layers.append(layers.MaxPooling1D(pool_size=pool_size))
    
            self.time4 = layers.Conv1D(filters=filter_1_size*(2**(n_stacks+1)), kernel_regularizer=regularizer,
                                       kernel_size=kernel_size, padding='same')
            self.leakyrelu3 = layers.LeakyReLU(alpha=alpha)
            self.global_pooling = layers.GlobalAveragePooling1D()

        self.dense1 = layers.Dense(self.model_config.baseline_model.dense_layer_units)
        self.leakyrelu4 = layers.LeakyReLU(alpha=alpha)
        self.dense2 = layers.Dense(self.model_config.baseline_model.dense_layer_units)
        self.leakyrelu5 = layers.LeakyReLU(alpha=self.model_config.baseline_model.alpha)
        self.dense_out = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        if self.ds_type == 'soilnet':
            features, sample_indices = inputs
            x = graph_reshape([features, sample_indices], self.batch_size, self.timestep_before,
                        self.timestep_after, self.input_feature_numb, self.freq)  
        elif self.ds_type == 'cml':
            x = inputs
            
        x1 = self.time1(x)
        if self.model_config.baseline_model.type == 'cnn':
            x1 = self.leakyrelu1(x1)
        x1 = self.time2(x1)
        if self.model_config.baseline_model.type == 'cnn':
            x1 = self.leakyrelu2(x1)
        x1 = self.max_pooling(x1)
            
        for i in range(len(self.pooling_layers)):
            x1 = self.time_layers[2*i](x1)
            if self.model_config.baseline_model.type == 'cnn':
                x1 = self.leakyrelu_layers[2*i](x1)
            x1 = self.time_layers[2*i+1](x1)
            if self.model_config.baseline_model.type == 'cnn':
                x1 = self.leakyrelu_layers[2*i+1](x1)
            x1 = self.pooling_layers[i](x1)
        x1 = self.time4(x1)
        if self.model_config.baseline_model.type == 'cnn':
            x1 = self.leakyrelu3(x1)
            x1 = self.global_pooling(x1)

        x1 = self.dense1(x1)
        x1 = self.leakyrelu4(x1)
        x1 = self.dense2(x1)
        x1 = self.leakyrelu5(x1)
        out = self.dense_out(x1)
        return out