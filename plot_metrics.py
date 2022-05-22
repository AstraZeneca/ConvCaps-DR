
import numpy as np   
import datetime  
import matplotlib.pyplot as plt 


def plot_metrics_TrnVal(x_array, trn_metric, val_metric, val_metric2=None, 
                        title='Loss', xlabel='Iterations', ylabel='Batch error', 
                        ylim=None, legend=['Training', 'Validation'], 
                        save_folder=None, check_best=None):
    """ It plots a metric, training and validation, and saves the image.
    By default, it has the names for the losses. A third set can be introduced.
    """
    fig = plt.figure(figsize=(20,10))
    plt.plot(x_array, trn_metric, 'b')
    plt.plot(x_array, val_metric, 'r')
    if val_metric2 is not None:
        plt.plot(x_array, val_metric2, 'g')
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(legend, title='Image type', fontsize=15)  
    if ylim is None:
        ylim = np.nanmax([np.nanmax(trn_metric), np.nanmax(val_metric)])  
        plt.ylim([0,ylim])
    else:
        plt.ylim(ylim)    
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='1.0', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=5)
    if save_folder is not None:
        fileOut = save_folder + title + "_Validation.png" 
        plt.savefig(fileOut) #save the figure in a file
    #plt.show()  
    plt.clf()
    plt.close(fig)
    if check_best is not None and len(trn_metric) > 3:
        best_metric = val_metric if val_metric2 is None else val_metric + \
                                                             val_metric2
        ix_model = np.argsort(best_metric)
        if check_best == 'max':
            ix_model = np.flip(ix_model)
        print('The best 3 models based on ' + title + ' are: ')
        print('   ' + str(1+ix_model[0]) + ': ' + str(best_metric[ix_model[0]]))
        print('   ' + str(1+ix_model[1]) + ': ' + str(best_metric[ix_model[1]]))
        print('   ' + str(1+ix_model[2]) + ': ' + str(best_metric[ix_model[2]]))



def print_metrics_model(metrics, stepBatch=0, typeSet='Training', num=1):
    """ It prints the last values in *metrics* """
    if num == 1:
        val0 = metrics[0,-1]
        val1 = metrics[1,-1]
    else:  
        end = metrics.shape[1]
        val0 = np.mean(metrics[0,-num:end])
        val1 = np.mean(metrics[1,-num:end])      
    print('   ' + typeSet + ', crossEntropy-loss ' + 
          'at step %d: %.4f' % (stepBatch, val0))
    print('   ' + typeSet + ', accuracy: %.4f' % val1)   
  


def print_time_cost(timeCur, timeIni=None, iCurr=None, iNmbr=None, eCurr=1,  
                    eNmbr=None, it2ev=1, strT='training', flagFuture=False):
    """ Given the time at the beginning of the cycle (timeCur), it prints the 
    time spent in that cycle. Optionally, it prints the time since the 
    beginning of the training (timeIni) and the expected future time.
    All times are in datetime format.
        - timeCur: The time at the beginning of the current iteration
        - timeIni: The time at the beginning of the training
        - iCurr: The current iteration (within the current epoch)
        - iNmbr: The total number of iterations in one epoch.
        - eCurr: The current epoch
        - eNmbr: The total number of epochs.
    """ 
    OneIter = datetime.datetime.now() - timeCur
    print('Epoch '+ str(eCurr) +'. Preparing the ' + strT + 
          ' took: ', OneIter, ' seconds')
    if flagFuture:
        PastTime = datetime.datetime.now() - timeIni
        FutrTime = (iNmbr - iCurr + iNmbr*(eNmbr-eCurr) ) * OneIter / it2ev
        print('   Past   time:  ', PastTime, ' seconds')
        print('   Future time:  ', FutrTime, ' seconds')
        print('   Total  time:  ', PastTime + FutrTime, ' seconds')