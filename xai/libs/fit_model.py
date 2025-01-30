from .metrics import mcc_metric, matthews_correlation
import tensorflow as tf
import numpy as np
import wandb, os
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,  LearningRateScheduler


def calculate_weights(model_config, train_dataset_batched):
    if model_config.weight_classes.use:
        if model_config.weight_classes.calculate:
            # iterate training dataset and count the number on anonamlies
            anomaly_lists = []
            for batch in train_dataset_batched:
                anomaly_lists.append(batch[1])
            anomaly_lists = np.concatenate(anomaly_lists)
            total_numb = len(anomaly_lists)
            anomaly_numb = sum(anomaly_lists) 
            classes_weights = {0: total_numb/(total_numb - anomaly_numb), 1: 2*total_numb/anomaly_numb}
        elif (model_config.weight_classes.class_0 is not None) and (model_config.weight_classes.class_1 is not None):
            classes_weights = {0: model_config.weight_classes.class_0, 1: model_config.weight_classes.class_1}
        else:
            classes_weights = {0: 1, 1: 50}
    else:
        classes_weights = None
    print('classes weights: ', classes_weights)
    return classes_weights
        

class MCC_custom(tf.keras.callbacks.Callback):

    def __init__(self, train, validation=None):
        super(MCC_custom, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):
        logs['MCC_score_train'] = float('-inf')
        y_train = []
        y_train_pred = []
        for x in self.train:
            y_train.append(x[1])
            y_train_pred.append(self.model(x[0], training=False))
        y_train = np.concatenate(y_train)
        y_train_pred = np.concatenate(y_train_pred)
        score = mcc_metric(y_train, y_train_pred)       
        if (self.validation):
            logs['MCC_score_val'] = float('-inf')
            y_val = []
            y_val_pred = []
            for x in self.validation:
                y_val.append(x[1])
                y_val_pred.append(self.model(x[0], training=False))
            y_val = np.concatenate(y_val)
            y_val_pred = np.concatenate(y_val_pred)
            val_score = mcc_metric(y_val, y_val_pred)
            logs['MCC_score_train'] = np.round(score, 5)
            logs['MCC_score_val'] = np.round(val_score, 5)
        else:
            logs['MCC_score_train'] = np.round(score, 5)


def train_model(model, model_config, preproc_config, train_dataset_batched, val_dataset_batched=None, 
                baseline=False, classes_weights=None, CV=False, labels=None, split_numb=None):
    hyperparameters_preprocessing ={'batch_size': preproc_config.batch_size,
                                    'timestep_before': preproc_config.timestep_before,
                                    'timestep_after': preproc_config.timestep_after}
    config_dictionary = {**hyperparameters_preprocessing, **preproc_config, **model_config,
                         'classes_weights': classes_weights, 'split_numb': split_numb}
    if baseline == False:
        wandb.init(project=model_config.wandb_project, config=config_dictionary, tags=labels)
    else:
        wandb.init(project=model_config.baseline_model.wandb_project, config=config_dictionary, tags=labels)
    print(wandb.run.dir)
    optimizer = {'adam': tf.keras.optimizers.Adam(model_config.learning_rate),
                 'sgd': tf.keras.optimizers.SGD(model_config.learning_rate),
                 'rmsprop': tf.keras.optimizers.RMSprop(model_config.learning_rate)
    }.get(model_config.optimizer)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer, 
        metrics=[tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.TruePositives(name='tp'),
                 tf.keras.metrics.FalsePositives(name='fp'),
                 tf.keras.metrics.TrueNegatives(name='tn'),
                 tf.keras.metrics.FalseNegatives(name='fn')],
        )
    if CV == False:
        es = EarlyStopping(monitor='val_loss', patience=model_config.es_patience, restore_best_weights=True)
        mc = ModelCheckpoint(os.path.join(wandb.run.dir, 'model'), mode='min', save_best_only=True)
    else:
        es = EarlyStopping(monitor='loss', patience=model_config.es_patience, restore_best_weights=True)
        mc = ModelCheckpoint(os.path.join(wandb.run.dir, 'model'), monitor='loss', mode='min', save_best_only=True)
    if model_config.graph_convolution.layer!='GATConv':
        wandb_callback = WandbCallback(save_model=True)
    else:
        if CV == False:
            mc = ModelCheckpoint(os.path.join(wandb.run.dir, 'model'), monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
        else:
            mc = ModelCheckpoint(os.path.join(wandb.run.dir, 'model'), monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
        #wandb_callback = [WandbMetricsLogger(), WandbModelCheckpoint(wandb.run.dir, save_weights_only=True)]
        wandb_callback = WandbCallback(save_model=False)
        #wandb_callback = [WandbMetricsLogger()]
        
    callback_list = [wandb_callback, PlotLossesKeras(), es, mc]
    classes_weights = calculate_weights(model_config, train_dataset_batched)
    
    if model_config.learning_learn_scheduler.use:
        def scheduler(epoch, lr):
            if epoch < model_config.learning_learn_scheduler.after_epochs:
                return lr 
            else:
                return lr * model_config.learning_learn_scheduler.rate     
        callback_list.append(LearningRateScheduler(scheduler))

    history = model.fit(
        train_dataset_batched,
        validation_data=val_dataset_batched,
        epochs=model_config.epochs,
        verbose=1,
        callbacks=callback_list, 
        class_weight = classes_weights,
        )
    wandb.finish()
    return history, model

