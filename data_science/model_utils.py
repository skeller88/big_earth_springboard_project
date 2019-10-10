import gc
import os

from data_science.augmented_image_sequence import AUGMENTATIONS_TEST, AugmentedImageSequence

should_train_model = True



def train_model_for_fold(num_fold, should_train_model):
    if should_train_model:
        train_index = indexes[num_fold][0]
        valid_index = indexes[num_fold][1]

        histories = train_fold(num_fold=num_fold, model=model, model_preprocess_func=model_preprocess_func,
                               x_train=x[train_index], y_train=y[train_index], x_valid=x[valid_index],
                               y_valid=y[valid_index], n_epochs=n_epochs, n_classes=n_classes,
                               n_layers_to_freeze=n_layers_to_freeze, batch_size=batch_size, random_state=random_seed,
                               weight_dir=weight_dir)

        full_history = defaultdict(list)
        for history in histories:
            for key, value in history.history.items():
                full_history[key] += value

        graph_model_history(full_history)
        return history


def graph_model_history(history):
    # history: List[Dict]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('Basic CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    max_epoch = len(history['acc'])+1
    epoch_list = list(range(1,max_epoch))
    ax1.plot(epoch_list, history['acc'], label='Train Accuracy')
    ax1.plot(epoch_list, history['val_acc'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(1, max_epoch, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(1, max_epoch, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")



def train_fold(num_fold, model, model_preprocess_func, x_train: np.array, y_train: np.array, x_valid: np.array,
               y_valid: np.array, n_epochs, n_classes, n_layers_to_freeze, batch_size, random_state, weight_dir):
    """
    Based on from https://www.kaggle.com/infinitewing/keras-solution-and-my-experience-0-92664
    """

    print(f'Start KFold number {num_fold}')
    print(f'Split train: {len(x_train)}')
    print(f'Split valid: {len(x_valid)}')

    weight_path = os.path.join(weight_dir, 'weights_kfold_' + str(num_fold) + '.h5')
    threshold_path = os.path.join(weight_dir, 'thresholds_kfold_' + str(num_fold) + '.csv')

    histories = []
    learn_rates = [0.001, 0.0001, 0.00001]
    for learn_rate_num, learn_rate in enumerate(learn_rates):
        print(f'Training model on fold {num_fold} with learn_rate {learn_rate}')
        opt = Adam(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        #             logdir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001),
                     #                          TensorBoard(logdir, histogram_freq=1)
                     ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]

        # Generators
        train_generator = AugmentedImageSequence(x=x_train, y=y_train, batch_size=batch_size,
                                                 model_preprocess_func=model_preprocess_func,
                                                 augmentations=AUGMENTATIONS_TRAIN)

        valid_batch_size = int(batch_size * .75)
        valid_generator = AugmentedImageSequence(x=x_valid, y=y_valid, batch_size=valid_batch_size,
                                                 model_preprocess_func=model_preprocess_func,
                                                 augmentations=AUGMENTATIONS_TEST)

        history = model.fit_generator(generator=train_generator,
                                      epochs=n_epochs,
                                      steps_per_epoch=len(train_generator),
                                      callbacks=callbacks,
                                      validation_data=valid_generator, validation_steps=len(valid_generator),
                                      shuffle=True, verbose=1)
        histories.append(history)
        # Fine-tune the base model after the final iteration using a low learning rate and SGD
        # Looks like SGD is slower than ADAM, but best for generalization performance:
        # https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/
        # https://stackoverflow.com/questions/47626254/changing-optimizer-in-keras-during-training?rq=1

        if learn_rate_num + 1 == len(learn_rates):
            print('Fine tuning the model one last time.')
            for layer in model.layers[n_layers_to_freeze:]:
                layer.trainable = True

            optimizer = RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
            #             optimizer = SGD(lr=learn_rate, momentum=0.9)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit_generator(generator=train_generator,
                                          epochs=n_epochs,
                                          steps_per_epoch=len(train_generator),
                                          callbacks=callbacks,
                                          validation_data=valid_generator, validation_steps=len(valid_generator),
                                          shuffle=True, verbose=1)
            histories.append(history)

    # data_valid will be shuffled at this point by the valid_generator, so create a new generator
    predict_generator = AugmentedImageSequence(x=x_valid, y=None, batch_size=valid_batch_size,
                                               model_preprocess_func=model_preprocess_func,
                                               augmentations=AUGMENTATIONS_TEST)
    pred_y_valid = model.predict_generator(predict_generator, steps=len(predict_generator))

    # Fine tune the class prediction thresholds by brute force on the validation data set
    # https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
    threshold = get_optimal_class_threshold(y_valid, pred_y_valid)

    # Store thresholds
    pd.DataFrame(data=threshold).transpose().to_csv(threshold_path, index=False)

    # Attempt to avoid memory leaks
    del train_generator
    del valid_generator
    del predict_generator
    gc.collect()

    return histories


def kfold_predict(model, model_preprocess_func, weight_dir, x_test, batch_size, classes, nfolds):
    """
    Average predictions across all folds. From https://www.kaggle.com/infinitewing/keras-solution-and-my-experience-0-92664
    """
    sum_pred_test_probs = np.zeros((len(x_test), len(classes)))
    sum_thresholds = np.zeros(len(classes))
    for num_fold in range(0, nfolds):
        print(f'Predicting with KFold number {num_fold}')
        thresholds_path = os.path.join(weight_dir, 'thresholds_kfold_' + str(num_fold) + '.csv')
        sum_thresholds += pd.read_csv(thresholds_path).iloc[0].values

        #         weight_path = os.path.join(weight_dir, 'weights_kfold_' + str(num_fold) + '.h5')
        #         model.load_weights(weight_path)

        predict_generator = AugmentedImageSequence(x=x_test, y=None, batch_size=batch_size,
                                                   model_preprocess_func=model_preprocess_func,
                                                   augmentations=AUGMENTATIONS_TEST)

        # Generators
        sum_pred_test_probs += model.predict_generator(predict_generator)

        del predict_generator
        gc.collect()

    return sum_pred_test_probs / nfolds, sum_thresholds / nfolds


def predict_for_weights(weight_dir):
    df = pd.read_csv(dirname + '/train_v2.csv')
    df_test = df.iloc[n_samples:].copy()
    df_test['image_path'] = train_path + '/' + df_test['image_name'] + '.jpg'
    df_test['split_tags'] = df_test['tags'].map(lambda row: row.split(" "))
    lb = MultiLabelBinarizer()
    labels = lb.fit_transform(df_test['split_tags'])

    avg_pred_test_probs, avg_thresholds = kfold_predict(model, weight_dir=weight_dir, x_test=df_test['image_path'].values, classes=classes, nfolds=nfolds, batch_size=batch_size)

    # Generate the labels
    pred_test_labels = pd.DataFrame(avg_pred_test_probs, columns=classes)
    pred_test_labels = pred_test_labels.apply(lambda x: x > avg_thresholds, axis=1)
    # Convert boolean predictions to labels
    pred_test_lables = pred_test_labels.apply(lambda row: ' '.join(row[row].index), axis=1)
    return labels, pred_test_labels