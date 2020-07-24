import cv2
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
import skimage.feature
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

''''
Instructions: 
    - the main is at the bottom of the page
    - class_indexes + data_path is at the first 2 code rows 
'''


def GetDefaultParameters():
    '''
        initiating the pipe parameters dictionary
    '''

    path = 'C:\\Users\\idofi\\OneDrive\\Documents\\BGU\\masters\\year A\\computer vision\\Task_1\\101_ObjectCategories'
    class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # default
    train_size = 20
    test_size = 20
    train_ratio = 0.5  # if needed
    # Best configuration
    image_size = 70
    HOG_orientation_bins = 14
    HOG_cell_size = (8, 8)
    HOG_cells_per_block = (2, 2)
    block_norm = 'L2'
    SVM_c = 1.8
    SVM_gamma = (1 / 2744)
    SVM_degree = 2
    SVM_kernel = 'rbf'

    params = {'Data': {'path': path, 'class_indexes': class_indexes},
              'Split': {'train_size': train_size, 'test_size': test_size, 'train_ratio': train_ratio},
              'Prepare': {'size': image_size,
                          'HOG': {'orientation_bins': HOG_orientation_bins, 'cell_size': HOG_cell_size,
                                  'cells_per_block': HOG_cells_per_block, 'block_norm': block_norm}},
              'Model': {'SVM': {'kernel': SVM_kernel, 'gamma': SVM_gamma, 'degree': SVM_degree, 'C': SVM_c}}}

    return params


def GetData(data_params, saveToPkl):
    '''
    creating dataframe with the images data
    :param  data_params['path']: path to 101_ObjectCategories folder
    :param  data_params['class_indexes]: the classes index for the pip
    :param  saveToPkl: serializing to pickle file (boolean)
    :return dataframe
    '''
    # setting class names for indexes
    classes_array = []
    for folder in os.listdir(data_params['path']):
        classes_array.append(folder)

    ##Sorting folders alphabeticlly
    classes_array = sorted(classes_array,key=str.lower)


    print(' ---- Importing data ---- ')
    raw_data = pd.DataFrame(columns=['Data', 'Labels'])
    for i in data_params['class_indexes']:
        images_arr = os.listdir('{}\\{}'.format(data_params['path'], classes_array[i]))
        sorted_images = sorted(images_arr)
        for image in sorted_images:
            readed_image = cv2.imread('{}\\{}\\{}'.format(data_params['path'], classes_array[i], image))
            # grey scale the images
            gray_image = cv2.cvtColor(readed_image, cv2.COLOR_BGR2GRAY)
            # apply image smoothing using GaussianBlur
            blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
            temp_df = pd.DataFrame({'Data': [blur], 'Labels': i})
            raw_data = raw_data.append(temp_df, ignore_index=True)

    if saveToPkl:
        raw_data.to_pickle('fold1_raw_data.pkl')

    return raw_data


def TrainTestSplit(DandL, Params):
    '''
    splits the data into train and test
    :param DandL: raw dataframe
    :param Params
    :return: TrainData & TestData (in dataframe format)
    '''

    TrainData = pd.DataFrame()
    TestData = pd.DataFrame()

    for i in Params['Data']['class_indexes']:
        data_i = DandL[DandL['Labels'] == i]
        # take the first 20 images
        trainData = data_i[:20]
        # store the rest of the images in the i class
        restOfData = data_i.iloc[~data_i.index.isin(trainData.index)]
        if len(restOfData) < 20:  # in case there are less then 20 images left
            testData = restOfData
            print('class {} has {} test samples\n'.format(i, len(restOfData)))
        else:
            testData = restOfData[:20]

        TrainData = TrainData.append(trainData, ignore_index=True)
        TestData = TestData.append(testData, ignore_index=True)

    return TrainData, TestData


def resize_data(data, Params):
    '''
    resize the images into SxS dimension according to Params
    :param s_param: the new images size
    :param: data: the dataframe
    :return: new dataframe with resized images
    '''
    copy_data = data.copy()
    size = Params['size']
    copy_data['Data'] = copy_data['Data'].apply(lambda x: cv2.resize(x, (size, size)))
    return copy_data


def prepare(data, Params):
    '''
       applying HOG on the raw_data using the defined Params value, converting the images into features vector
       :param:data: image + labels dataframe
       :param: Params: pipe parameters
       :return: Two numpy arrays of feature vectors and labels
       '''

    data_copy = data.copy()
    # resize images
    data_copy = resize_data(data_copy, Params)

    hog_features = []
    labels = []

    # apply HOG on every row in the dataframe
    for row in data_copy.itertuples():
        hog_rep, hog_vis = skimage.feature.hog(row.Data, orientations=Params['HOG']['orientation_bins'],
                                               pixels_per_cell=Params['HOG']['cell_size'],
                                               cells_per_block=Params['HOG']['cells_per_block'],
                                               block_norm=Params['HOG']['block_norm'], visualize=True,
                                               transform_sqrt=False, feature_vector=True)

        label = row.Labels

        hog_features.append(hog_rep)
        labels.append(label)

    # use standard normalization to normalize the feature using their mean and variance
    scalar = StandardScaler()
    scalar.fit(hog_features)
    hog_features = scalar.transform(hog_features)
    return np.array(hog_features), np.array(labels)


def SVM_train(train_x, train_y, Params):
    '''
      fit SVM classifier (used for linear kernel only)
      :param train_x: training feature vectors
      :param: train_y: training labels
      :param: Params: pipe parameters
      :return: trained classifier
    '''

    # declare the classifier
    SVM = svm.SVC(kernel=Params['Model']['SVM']['kernel'],
                  C=Params['Model']['SVM']['C'],
                  gamma=Params['Model']['SVM']['gamma'],
                  degree=Params['Model']['SVM']['degree'],
                  decision_function_shape='ovr',
                  probability=False)

    SVM.fit(train_x, train_y)

    return SVM


def findWorstPredictions(predictions, TestDataRep_y, decision_function, Params):
    '''
    Finding the worst error images in each class (Max amount of 2 images in each class)
    :param   predictions: Vector of predicitons to test set
    :param   TestDataRep_y: Vector of True clsses to test set
    :param   decision_function: A matrix of shape (n_test_set,n_classes) with decision_function to each sample.
    :param   Params: Vector with all model parameters
    :return preds_and_func: Data frame with the worst images prediction indexes in the test(Max 2 in each class)
    '''

    preds_and_func = pd.DataFrame()
    total_scores = []
    for i in range(0, len(decision_function)):
        # get the index of the i class given the class number
        class_index = Params['Data']['class_indexes'].index(TestDataRep_y[i])
        total_scores.append(decision_function[i][class_index] - max(decision_function[i]))

    preds_and_func['predictions'] = predictions
    preds_and_func['Real_class'] = TestDataRep_y
    preds_and_func['decision_function'] = total_scores

    indxs_of_worst_ims = []
    # Removing all correct predictions
    indexNames = preds_and_func[(preds_and_func['predictions']) == (preds_and_func['Real_class'])].index
    preds_and_func.drop(indexNames, inplace=True)
    preds_and_func = preds_and_func.sort_values(['Real_class', 'decision_function'], ascending=True)
    amount_appeard = 0
    for class_i in Params['Data']['class_indexes']:
        amount_apeared = 0
        for idx, row in preds_and_func.iterrows():

            curr_class = row['Real_class']
            if class_i == curr_class:
                if amount_apeared < 2:
                    amount_apeared = amount_apeared + 1
                    indxs_of_worst_ims.append(idx)

                else:
                    preds_and_func = preds_and_func.drop(index=idx)

    return preds_and_func


def SVM_test(Model, TestDataRep_x, TestDataRep_y, Params):
    '''
      predict classes using the train classifier in óne-vs-rest mode and
      :param Model: trained classifier
      :param TestDataRep_x: testing feature vectors
      :param: TestDataRep_y: testing labels
      :param: Params: pipe parameters
      :return: worst classified images + predictions vector
    '''
    predictions = Model.predict(TestDataRep_x)
    dec_func = Model.decision_function(TestDataRep_x)
    preds_and_func = findWorstPredictions(predictions, TestDataRep_y, dec_func, Params)

    return preds_and_func, predictions


def evaluate(Results, TestDataRep_y, Params):
    '''
         Claculate the model error and creating confusion matrix
         :param Results: predictions vector
         :param TestDataRep_y: real values vector
         :param: Params: pipe parameters
         :return: dictionary with the error rate and the confusion matrix
   '''

    error_rate = round(1 - accuracy_score(TestDataRep_y, Results), 3)
    class_indexes = Params['Data']['class_indexes']
    conf_matrix = confusion_matrix(TestDataRep_y, Results, labels=class_indexes)

    summary = {'error_rate': error_rate, 'conf_matrix': conf_matrix}

    return summary


def ReportResults(Summary, preds_and_func, TestData, Params, displayImages):
    '''
    printing the sumarry of results

    :param   Summary: Data frame with train error,validation error,Confusion matrix
    :param   preds_and_func: Data frame with the worst images prediction indexes(Max 2 in each class)
    :param   TestData: All images in the test set
    :param   Params: Vector with all model parameters
    :param   displayImages: bool. if true: figure of worst images prediction will appear on the screen

    :return None

    '''

    print('\n')
    print('|---------------------------------------|')
    print('|-------------|  Results  |-------------|')
    print('|---------------------------------------|')
    print('\nModel Configuration: ')

    size = Params['Prepare']['size']
    bins = Params['Prepare']['HOG']['orientation_bins']
    cell_size = Params['Prepare']['HOG']['cell_size']
    cells_block = Params['Prepare']['HOG']['cells_per_block']
    norm = Params['Prepare']['HOG']['block_norm']
    kernel = Params['Model']['SVM']['kernel']
    c = Params['Model']['SVM']['C']
    gamma = Params['Model']['SVM']['gamma']
    degree = Params['Model']['SVM']['degree']

    print('\nkernel:{} img_size:{} orientation_bins:{} cell_size:{}'.format(kernel, size, bins, cell_size))
    print('cells_per_block:{} block_norm:{}  C:{} Gamma:{} '.format(cells_block, norm, c, gamma))
    print('\nConfusion Matrix: \n{}'.format(Summary['conf_matrix']))
    print('\nTest Error: {}'.format(Summary['error_rate']))

    # present worst classified images of each class
    if displayImages:
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Worst two classified images for each class', fontsize="x-large")
        loc = 1
        for idx in preds_and_func.index:
            df = preds_and_func.loc[[idx]]
            fig.add_subplot(3, 7, loc)
            loc = loc + 1
            imsize = Params['Prepare']['size']
            mig = plt.imshow(cv2.resize(TestData.iloc[idx][0], (imsize, imsize)), cmap='gray')
            tit = 'True:{} Pred:{} \n Score{}'.format(df.iat[0, 1], df.iat[0, 0], round(df.iat[0, 2], 2))
            plt.title(tit, fontsize=9)
            mig.axes.get_xaxis().set_visible(False)
            mig.axes.get_yaxis().set_visible(False)

        plt.show()


def n_class_SVM_train(train_x, train_y, Params):
    '''
       "one-vs-rest" implementation - fitting 10 classifiers

       :param   train_x: features vector of training data
       :param   train_y: labels vector of training data
       :param   Params: Vector with all model parameters

       :return vector of 10 trained classifiers

       '''
    n_class_SVM = []

    for class_i in Params['Data']['class_indexes']:
        temp_train_y = train_y.copy()

        # declare the classifier
        SVM_i = svm.SVC(kernel=Params['Model']['SVM']['kernel'],
                        C=Params['Model']['SVM']['C'],
                        gamma=Params['Model']['SVM']['gamma'],
                        degree=Params['Model']['SVM']['degree'])

        # change class_i label to 1 and "the-rest" to -1
        for idx, label in enumerate(temp_train_y):
            if label == class_i:
                temp_train_y[idx] = 1
            else:
                temp_train_y[idx] = -1

        # train the model
        SVM_i.fit(train_x, temp_train_y)

        n_class_SVM.append(SVM_i)

    return n_class_SVM


def n_class_SVM_predict(MClassSVM, TestDataRep_x, TestDataRep_y, Params):
    '''
       "one-vs-rest" implementation - perform classification with each trained svm
       and chooses the most "certain" one according the distance function

       :param   MClassSVM: vector of 10 trained classifiers
       :param   TestDataRep_x: test features vector
       :param   TestDataRep_y: test labels vector
       :param   Params: Vector with all model parameters

       :return dataframe with the worst classified images and predictions vector

   '''

    score_results = []
    predictions = []

    # iterate on each image in the training set
    for idx, sample in enumerate(TestDataRep_x):
        scores = []
        # get prediction for each SVM classifier
        for class_idx, svm_i in enumerate(MClassSVM):
            # get distance from margin for current sample
            class_score = svm_i.decision_function([sample])
            # store the 10 distances in array
            scores.append(class_score[0])

        # using for getting the worst images
        score_results.append(scores)
        # find the predicted class out of 10 distances values
        predicted_class = Params['Data']['class_indexes'][np.argmax(scores)]
        predictions.append(predicted_class)

    # finding the worst predictions
    preds_and_func = findWorstPredictions(predictions, TestDataRep_y, score_results, Params)

    return preds_and_func, predictions


def annot_max(x, y, ax=None):
    '''
    Drawing the maximum/minimum on a plot
    :param x: Vector to x_axis
    :param y: Vector to y_axis
    :param ax: figure{not in use}
    '''

    xmax = x[np.argmin(y)]
    ymax = y.min()
    text = "Minimum error:img size ={:.3f}, Error={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=270")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="center")
    ax.annotate(text, fontsize=9, xy=(xmax, ymax), xytext=(0.5, 0.5), **kw)


def linear_hyperTuning_table(TrainData, Params, kernel, img_size_r, orientation_bins_r, cell_size_r,
                             cells_per_block_r, block_norm_r, C_r, gamma_r, degree_r):
    '''
     performing k-fold cross-validation for hyper parameters tuning over the linear kernel params
     the function gets possible ranges of hyper parameters and evaluate each configuration
     using both training error and validation error

     :param   TrainData: vector of 10 trained classifiers
     :param   Params: Vector with all model parameters
     :param   hyper-parameters lists

     :return dataframe with results for each configuration of hyper parameters

  '''

    columns = ['kernel', 'img_size', 'orientation_bins', 'cell_size', 'cells_per_block', 'block_norm', 'C', 'gamma',
               'degree', 'train_error', 'validation_error']
    main_tune_results = pd.DataFrame(columns=columns)

    # iterate over all possible configurations
    for size in img_size_r:
        for bins in orientation_bins_r:
            for cell_size in cell_size_r:
                for cells_block in cells_per_block_r:
                    for norm in block_norm_r:
                        for c in C_r:
                            for gamma in gamma_r:
                                for degree in degree_r:
                                    new_params = Params.copy()
                                    new_params['Prepare']['size'] = size
                                    new_params['Prepare']['HOG']['orientation_bins'] = bins
                                    new_params['Prepare']['HOG']['cell_size'] = cell_size
                                    new_params['Prepare']['HOG']['cells_per_block'] = cells_block
                                    new_params['Prepare']['HOG']['block_norm'] = norm
                                    new_params['Model']['SVM']['kernel'] = kernel
                                    new_params['Model']['SVM']['C'] = c
                                    new_params['Model']['SVM']['gamma'] = gamma
                                    new_params['Model']['SVM']['degree'] = degree

                                    print(
                                        '\nkernel:{} img_size:{} orientation_bins:{} cell_size:{}'.format(kernel,
                                                                                                          size,
                                                                                                          bins,
                                                                                                          cell_size))

                                    print('cells_per_block:{} block_norm:{}  C:{} Gamma:{} degree:{}'.format(
                                        cells_block, norm, c, gamma, degree))
                                    # data representation
                                    TrainDataRep_x, TrainDataRep_y = prepare(TrainData, new_params['Prepare'])

                                    # apply cross-validation and get training and validation error
                                    cv_results = cross_validate(svm.SVC(C=c, kernel='linear'), TrainDataRep_x,
                                                                y=TrainDataRep_y,
                                                                return_train_score=True)

                                    print(
                                        'Validation Error: {}  Train Error: {}'.format(
                                            1 - np.mean(cv_results['test_score']),
                                            1 - np.mean(cv_results['train_score'])))
                                    main_tune_results = main_tune_results.append(
                                        {'kernel': kernel, 'img_size': size, 'orientation_bins': bins,
                                         'cell_size': cell_size,
                                         'cells_per_block': cells_block, 'block_norm': norm, 'C': c, 'gamma': gamma,
                                         'degree': degree, 'train_error': 1 - np.mean(cv_results['train_score']),
                                         'validation_error': 1 - np.mean(cv_results['test_score'])},
                                        ignore_index=True)

    return main_tune_results


def nonlinear_hyperTuning_table(TrainData, Params, kernel, img_size_r, orientation_bins_r, cell_size_r,
                                cells_per_block_r, block_norm_r, C_r, gamma_r, degree_r):
    '''
         performing stratified k-fold cross-validation for hyper parameters tuning over the non-linear kernel params
         the function gets possible ranges of hyper parameters and evaluate each configuration
         using both training error and validation error

          :param   TrainData: vector of 10 trained classifiers
          :param   Params: Vector with all model parameters
          :param   hyper-parameters lists

          :return dataframe with results for each configuration of hyper parameters

    '''

    columns = ['kernel', 'img_size', 'orientation_bins', 'cell_size', 'cells_per_block', 'block_norm', 'C', 'gamma',
               'degree', 'train_error', 'validation_error']
    main_tune_results = pd.DataFrame(columns=columns)

    # iterate over all possible configurations
    for size in img_size_r:
        for bins in orientation_bins_r:
            for cell_size in cell_size_r:
                for cells_block in cells_per_block_r:
                    for norm in block_norm_r:
                        for c in C_r:
                            for gamma in gamma_r:
                                for degree in degree_r:
                                    new_params = Params.copy()
                                    new_params['Prepare']['size'] = size
                                    new_params['Prepare']['HOG']['orientation_bins'] = bins
                                    new_params['Prepare']['HOG']['cell_size'] = cell_size
                                    new_params['Prepare']['HOG']['cells_per_block'] = cells_block
                                    new_params['Prepare']['HOG']['block_norm'] = norm
                                    new_params['Model']['SVM']['kernel'] = kernel
                                    new_params['Model']['SVM']['C'] = c
                                    new_params['Model']['SVM']['gamma'] = gamma
                                    new_params['Model']['SVM']['degree'] = degree

                                    print(
                                        '\nkernel:{} img_size:{} orientation_bins:{} cell_size:{}'.format(kernel,
                                                                                                          size,
                                                                                                          bins,
                                                                                                          cell_size))

                                    print('cells_per_block:{} block_norm:{}  C:{} Gamma:{} degree:{}'.format(
                                        cells_block, norm, c, gamma, degree))

                                    # data representation
                                    result_validation, result_train = KfoldCrossValidation(new_params, TrainData)
                                    print(
                                        'Validation Error: {}  Train Error: {}'.format(result_validation, result_train))
                                    # apply cross-validation and get training and validation error
                                    main_tune_results = main_tune_results.append(
                                        {'kernel': kernel, 'img_size': size, 'orientation_bins': bins,
                                         'cell_size': cell_size,
                                         'cells_per_block': cells_block, 'block_norm': norm, 'C': c, 'gamma': gamma,
                                         'degree': degree, 'train_error': result_train,
                                         'validation_error': result_validation},
                                        ignore_index=True)

    return main_tune_results


def KfoldCrossValidation(Params, TrainData):
    '''
         stratified k-fold cross-validation to create balanced folds since we have multiclasses problem

         :param   TrainData: vector of 10 trained classifiers
         :param   Params: Vector with all model parameters

         :return dataframe with results for each configuration of hyper parameters

    '''

    new_params = Params.copy()

    TrainDataRep_x, TrainDataRep_y = prepare(TrainData, new_params['Prepare'])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold = 1
    results_validation = []
    results_train = []

    # apply Cross-Validation using K-fold
    for train_index, test_index in skf.split(TrainDataRep_x, TrainDataRep_y):
        MClassSVM = n_class_SVM_train(TrainDataRep_x[train_index], TrainDataRep_y[train_index], new_params)

        # predict & evaluate on VALIDATION set
        preds_and_func, predictions_validation = n_class_SVM_predict(MClassSVM, TrainDataRep_x[test_index],
                                                                     TrainDataRep_y[test_index], new_params)

        Summary_validation = evaluate(predictions_validation, TrainDataRep_y[test_index], new_params)

        # predict & evaluate on TRAIN set
        preds_and_func, predictions_train = n_class_SVM_predict(MClassSVM, TrainDataRep_x[train_index],
                                                                TrainDataRep_y[train_index], new_params)

        Summary_train = evaluate(predictions_train, TrainDataRep_y[train_index], new_params)

        # append the  results into a list
        results_validation.append(Summary_validation['error_rate'])
        results_train.append(Summary_train['error_rate'])
        print(' Fold: {} -- validation error: {}  train error: {}'.format(fold, Summary_validation['error_rate'],
                                                                          Summary_train['error_rate']))

        fold = fold + 1

    # calc the avg error for both validation and training
    final_results_validation = np.mean(results_validation)
    final_results_train = np.mean(results_train)

    return final_results_validation, final_results_train


def tuning():
    '''
        main function to explore different parameters tuning
        we use the above functions to create plot of the tuning process
    '''

    np.random.seed(0)
    Params = GetDefaultParameters()

    class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # default
    Params['Data']['class_indexes'] = class_indexes

    # DandL = GetData(Params['Data'], True)
    DandL = pd.read_pickle('fold1_raw_data.pkl')

    TrainData, TestData = TrainTestSplit(DandL, Params)

    img_size_r = [70]
    orientation_bins_r = [14]
    cell_size_r = [(8, 8)]
    cells_per_block_r = [(2, 2)]
    block_norm_r = ['L1', 'L1-sqrt', 'L2', 'L2-Hys']
    # block_norm_r = ['L2']
    kernel = 'rbf'
    C_r = [1.8]
    gamma_r = [1 / 2744]  # auto
    degree_r = [2]

    hyperTuningTable = nonlinear_hyperTuning_table(TrainData, Params, kernel, img_size_r, orientation_bins_r,
                                                   cell_size_r, cells_per_block_r, block_norm_r, C_r, gamma_r, degree_r)

    pd.set_option('display.max_columns', 30)
    sorted_hyperTuningTable = hyperTuningTable.sort_values(['validation_error']).reset_index(drop=True)
    print(sorted_hyperTuningTable)

    # plotting
    # cell_size = [i[0] for i in hyperTuningTable['cell_size']]
    # formated_cell_size = ['{}x{}'.format(i, i) for i in cell_size]

    # block_sizes = [i[0] for i in hyperTuningTable['cells_per_block']]
    # formated_block_sizes = ['{}x{}'.format(i, i) for i in block_sizes]

    plt.plot(hyperTuningTable['block_norm'], hyperTuningTable['validation_error'], marker='o', color='b')
    plt.plot(hyperTuningTable['block_norm'], hyperTuningTable['train_error'], marker='o', color='g')
    plt.xlabel('block_norm')
    plt.ylabel('Error rate')
    plt.title("block normalization tuning -- kernel:rbf", fontsize=12, fontweight=0, color='black')
    plt.legend(['validation', 'training'])

    # draw minimum value
    # annot_max(block_norm_r, hyperTuningTable['validation_error'])
    plt.show()


# Evaluation for classes 1-10
def main_fold1():
    np.random.seed(0)
    Params = GetDefaultParameters()

    class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # default
    Params['Data']['class_indexes'] = class_indexes

    DandL = GetData(Params['Data'], False)
    # DandL = pd.read_pickle('fold1_raw_data.pkl')

    TrainData, TestData = TrainTestSplit(DandL, Params)

    print('Data split shape:\ntrain: {} \ntest: {}'.format(TrainData.shape, TestData.shape))

    TrainDataRep_x, TrainDataRep_y = prepare(TrainData, Params['Prepare'])
    TestDataRep_x, TestDataRep_y = prepare(TestData, Params['Prepare'])

    # non linear SVM
    MClassSVM = n_class_SVM_train(TrainDataRep_x, TrainDataRep_y, Params)
    preds_and_func, predictions = n_class_SVM_predict(MClassSVM, TestDataRep_x, TestDataRep_y, Params)

    Summary = evaluate(predictions, TestDataRep_y, Params)

    ReportResults(Summary, preds_and_func, TestData, Params, False)


# Evaluation for classes 11-20
def main_fold2():
    np.random.seed(0)
    Params = GetDefaultParameters()
    # Choose classes number (Notice that class 1 notate as 0)
    class_indexes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    Params['Data']['class_indexes'] = class_indexes

    DandL = GetData(Params['Data'], False)
    # DandL = pd.read_pickle('fold1_raw_data.pkl')

    TrainData, TestData = TrainTestSplit(DandL, Params)

    print('Data split shape:\ntrain: {} \ntest: {}'.format(TrainData.shape, TestData.shape))

    TrainDataRep_x, TrainDataRep_y = prepare(TrainData, Params['Prepare'])
    TestDataRep_x, TestDataRep_y = prepare(TestData, Params['Prepare'])

    # non linear SVM
    MClassSVM = n_class_SVM_train(TrainDataRep_x, TrainDataRep_y, Params)
    preds_and_func, predictions = n_class_SVM_predict(MClassSVM, TestDataRep_x, TestDataRep_y, Params)

    Summary = evaluate(predictions, TestDataRep_y, Params)

    ReportResults(Summary, preds_and_func, TestData, Params, True)


# General Evaluation
def final_main(class_indexes_input, data_path_input):
    np.random.seed(0)

    # setting the pipe params
    Params = GetDefaultParameters()
    Params['Data']['class_indexes'] = class_indexes_input
    Params['Data']['path'] = data_path_input

    DandL = GetData(Params['Data'], False)

    TrainData, TestData = TrainTestSplit(DandL, Params)

    print('Data split shape:\ntrain: {} \ntest: {}'.format(TrainData.shape, TestData.shape))

    TrainDataRep_x, TrainDataRep_y = prepare(TrainData, Params['Prepare'])
    TestDataRep_x, TestDataRep_y = prepare(TestData, Params['Prepare'])

    # rbf SVM
    MClassSVM = n_class_SVM_train(TrainDataRep_x, TrainDataRep_y, Params)
    preds_and_func, predictions = n_class_SVM_predict(MClassSVM, TestDataRep_x, TestDataRep_y, Params)

    Summary = evaluate(predictions, TestDataRep_y, Params)

    ReportResults(Summary, preds_and_func, TestData, Params, True)


if __name__ == '__main__':
    class_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    data_path = ''  # please enter path to 101_ObjectCategories directory

    if data_path == '':
        print('Please insert path to image directory')
    else:
        final_main(class_indices, data_path)  # Task evaluation pipe (Pls use this one)

    # Only for modeling
    # tuning()  # for hyper parameters tuning
    # main_fold1()  # end-to-end modeling pipe
    # main_fold2()  # end-to-end evaluation pipe
