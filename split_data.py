
import numpy as np 
from sklearn.model_selection import KFold
from build_batches import load_image_names 
import csv


def separate_dataset_KFolds(img_name, test_fold, num_folds, flag_shuffle=True):
    """Separates the dataset into K-folds (num_folds), selecting one fold for  
    testing (test_fold) and the remaining for training.   
    The separation is done on a list of names. 
    """
    # Create the KFold generator
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=5)
    img_nums = np.arange(img_name.shape[0])     
    ix = 1  # Iterative index used to choose the test fold
    for trn_index, tst_index in kf.split(img_nums):
        if test_fold == ix:             
            trn_nam = img_name[trn_index] 
            tst_nam = img_name[tst_index]      
        ix = ix + 1
    # Shuffle the set
    if flag_shuffle:
        nTrn = trn_nam.shape[0]
        iTrn = np.random.permutation(nTrn)
        nTst = tst_nam.shape[0]
        iTst = np.random.permutation(nTst)
        trn_nam = trn_nam[iTrn]
        tst_nam = tst_nam[iTst]
    return trn_nam, tst_nam


# -----------------------------------------------------------------------------
def split_dataset_myeloid(args):
    """Separates the dataset Myeloid into training, validation, and testing.
    The separation will be printed on screen.
    UPDATE: In our experiments, we performed a 5-fold CV, so here we removed the 
    second split; thus, set validation and testing as the same.
    # Arguments (within args)
        class_folders:  list of str, the folders of the different classes
        num_folds:      int, the total number of folds to separate the data
        test_fold:      int, the index of the fold to use as test set
        vald_fold:      int, the index of the fold to use as validation set, in 
                        a second K-split
    # Returns
        trn_img:        list of str, the names of the training set (with path)
        val_img:        list of str, the names of the validation set (with path)
        tst_img:        list of str, the names of the test set (with path)
    """
    np.random.seed(42)   
    print('Loading images names from directory.....')
    class_names =  [0]*len(args.class_folders)
    trn_nam = [0]*len(args.class_folders)
    val_nam = [0]*len(args.class_folders)
    tst_nam = [0]*len(args.class_folders)
    trn_img = [0]*len(args.class_folders)
    val_img = [0]*len(args.class_folders)
    tst_img = [0]*len(args.class_folders)
    
    print('Splitting set (shuffled):')
    for ii in range(len(args.class_folders)):
        class_names[ii] = load_image_names(args.class_folders[ii], '.tiff') 
        trn_nam[ii], tst_nam[ii] = separate_dataset_KFolds(class_names[ii], 
            test_fold=args.test_fold, num_folds=args.num_folds) 
    #    trn_nam[ii], val_nam[ii] = separate_datasets_KFolds(trn_nam[ii], 
    #        test_fold=args.vald_fold, num_folds=args.num_folds-1)      
    val_nam = tst_nam   # Validation and testing will be the same  
    
    print(' \t\t\t\t\t\tTrain \t\tValid \t\tTest')
    for ii in range(len(args.class_folders)):
        print('  Class ', str(ii+1).zfill(2), '-', 
              args.class_descrip[ii], ' \t\t', 
              trn_nam[ii].shape[0], '\t\t', 
              val_nam[ii].shape[0], '\t\t', 
              tst_nam[ii].shape[0]) 
        trn_img[ii] = [args.class_folders[ii] + jj for jj in trn_nam[ii]]	
    #    val_img[ii] = [args.class_folders[ii] + jj for jj in val_nam[ii]]	  
        tst_img[ii] = [args.class_folders[ii] + jj for jj in tst_nam[ii]]	
    val_img = tst_img   # Validation and testing will be the same  
    
    nTrn, nVal, nTst = 0, 0, 0
    for ii in range(len(args.class_folders)):
        nTrn = nTrn + len(trn_nam[ii])
    #    nVal = nVal + len(val_nam[ii])
        nTst = nTst + len(tst_nam[ii])
    nVal = nTst        # Validation and testing will be the same  
    print('  Total \t\t\t\t\t', str(nTrn), ' \t', str(nVal), ' \t\t', str(nTst))
    print('')
    return trn_img, val_img, tst_img


# -----------------------------------------------------------------------------
def split_dataset_leukemia(args):
    """Separates the training set 'Leukemia' into training and validation, and 
    it loads the names of the test images and their labels.
    The separation will be printed on screen.
    # Arguments (within args)
        pos_folder:     str, the folder with the images of the positive class
        neg_folder:     str, the folder with the images of the negative class
        num_folds:      int, the total number of folds to separate the data
        vald_fold:      int, the index of the fold to use as validation set
    # Returns
        trnPos_img:     list of str, the file names of the positive class in the
                        training set (with path)
        trnNeg_img:     list of str, the file names of the negative class in the 
                        training set (with path)
        valPos_img, valNeg_img: list of str, same but for the validation set.
        tst_img:        list of str, the names of the test set (with path)
        tst_lbl:        numpy array (N,), the label of the test images.
    """
    np.random.seed(42)  
    print('Loading images names from directory.....')
    imgPos_names = load_image_names(args.pos_folder, '.png') 
    imgNeg_names = load_image_names(args.neg_folder, '.png') 

    trnPos_nam, valPos_nam = separate_dataset_KFolds(imgPos_names, 
                                                     test_fold=args.vald_fold, 
                                                     num_folds=args.num_folds)        
    trnNeg_nam, valNeg_nam = separate_dataset_KFolds(imgNeg_names, 
                                                     test_fold=args.vald_fold, 
                                                     num_folds=args.num_folds)   
    nPosTrn = trnPos_nam.shape[0]
    nPosVal = valPos_nam.shape[0]
    nNegTrn = trnNeg_nam.shape[0]
    nNegVal = valNeg_nam.shape[0]
    print('Split set (shuffled):')
    print('  Training:   ', nPosTrn,' Positives, ', nNegTrn,' Negatives.')
    print('  Validation: ', nPosVal,' Positives, ', nNegVal,' Negatives.')
    
    # Training (create the strings with 'path + file_names')
    trnNeg_img = np.ndarray(shape=(nNegTrn,), dtype=np.object)
    trnPos_img = np.ndarray(shape=(nPosTrn,), dtype=np.object)
    for ii in range(nPosTrn):
        trnPos_img[ii] = args.pos_folder + trnPos_nam[ii]
    for ii in range(nNegTrn):
        trnNeg_img[ii] = args.neg_folder + trnNeg_nam[ii]

    # Validation
    valNeg_img = np.ndarray(shape=(nNegVal,), dtype=np.object)
    valPos_img = np.ndarray(shape=(nPosVal,), dtype=np.object)
    for ii in range(nPosVal):
        valPos_img[ii] = args.pos_folder + valPos_nam[ii]
    for ii in range(nNegVal):
        valNeg_img[ii] = args.neg_folder + valNeg_nam[ii]

    # Test set. Load the test
    print('Loading test images.....')
    tst_nam = load_image_names(args.tst_folder, '.png') 
    nTst = len(tst_nam)
    print('  Test:       ', nTst,' images.')    
    with open(args.tst_labels, newline='') as f:
        reader = csv.reader(f)
        tst_file = list(reader)

    # Get the info for the test set. First row in tst_file is the header.
    # The order is not the same, so look at the names to set the label.
    tst_lbl = np.zeros(shape=(len(tst_nam),))
    for ii in range(len(tst_nam)):
        for jj in range(len(tst_file)):
            if tst_nam[ii][0:-4] == tst_file[jj][0][0:-4]:
                tst_lbl[ii] = int(tst_file[jj][2])
                break

    # Add the whole path to the names
    tst_img = np.ndarray(shape=(nTst,), dtype=np.object)
    for ii in range(nTst):
        tst_img[ii] = args.tst_folder + tst_nam[ii]

    return trnPos_img, trnNeg_img, valPos_img, valNeg_img, \
           tst_img, tst_lbl, tst_nam

