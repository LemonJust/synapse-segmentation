import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, accuracy_score, f1_score

def histogram():
    sns.set(style="white", palette="muted", color_codes=True)
    rs = np.random.RandomState(10)

    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    sns.despine(left=True)

    # Generate a random univariate dataset
    d = rs.normal(size=100)

    # Plot a simple histogram with binsize determined automatically
    sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

    # Plot a kernel density estimate and rug plot
    sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

    # Plot a filled kernel density estimate
    sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

    # Plot a historgram and kernel density estimate
    sns.distplot(d, color="m", ax=axes[1, 1])

    plt.setp(axes, yticks=[])
    plt.tight_layout()


def plot_confusion_matrix_v2(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.grid(b=None)
    plt.show()

def plot_confusion_matrix(y_true, y_pred,test_eval,
                          normalize=False,
                          title=None, axis = None, figure = None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = test_eval #'Confusion matrix'
    classes=['noise','syn.']
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if not figure:
        fig, ax = plt.subplots(figsize=(2.3,2.3),dpi=200)
    else: 
        fig = figure
        if test_eval == 'eval':
            cmap = plt.get_cmap('Greens')
            ax = axis[1]
        elif test_eval == 'test':
            cmap = plt.get_cmap('Blues')
            ax = axis[0]

    if test_eval == 'eval':
        cmap = plt.get_cmap('Greens')
    elif test_eval == 'test':
        cmap = plt.get_cmap('Blues')

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
       # print('Confusion matrix')

    #print(cm)

    
    bg = [[1,0],[0,1]]
    im = ax.imshow(bg, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontweight='bold',fontsize='medium',
                    color="white" if bg[i][j] > 0.5 else "black")
    fig.tight_layout()
    ax.grid(which='both')
    #return ax

def prob_to_pred(prob,cls_thr):
    if len(prob.shape) == 1:
        y_pred = prob>cls_thr       
    elif len(prob.shape) == 2: 

        if prob.shape[1]==1:
            y_pred = prob>cls_thr
        elif prob.shape[1]==2: 
            y_pred = np.argmax(prob,axis = 1)
    return y_pred

def test_and_eval(classifyre,X_test,y_test,X_eval,y_eval,cls_thr,report = False):

    #Predict the response for test dataset
    prob = classifyre.predict(X_test)
    y_pred_test = prob_to_pred(prob,cls_thr)

    #Predict the response for test dataset
    prob = classifyre.predict(X_eval)
    y_pred_eval = prob_to_pred(prob,cls_thr)

    if report :
        # Model Accuracy, how often is the classifier correct?
        print('Validation : ')
        print("Accuracy:",accuracy_score(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

        # Model Accuracy, how often is the classifier correct?
        print('Test : ')
        print("Accuracy:",accuracy_score(y_eval, y_pred_eval))
        print(classification_report(y_eval, y_pred_eval))
    fig, ax = plt.subplots(1,2, figsize=(5,2.3),dpi=200)
    f1_test = f1_score(y_test, y_pred_test)
    f1_eval = f1_score(y_eval, y_pred_eval)
    plot_confusion_matrix(y_test, y_pred_test, 'test',title = f'valid, f1 = {f1_test:.2f}', axis = ax, figure = fig)
    plot_confusion_matrix(y_eval, y_pred_eval, 'eval',title = f'test, f1 = {f1_eval:.2f}', axis = ax, figure = fig)
    return f1_test,f1_eval

