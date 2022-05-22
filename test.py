"""
FOR MYELOID DATASET:

"""
import os
import numpy as np
from build_batches import build_validation_batch
import sklearn.metrics as sk_metrics  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn 


def test_model_myeloid(model, tst_img, args):
    """ This function infers the classes for the test set and computes a series 
    of metrics. Specifically:
    - For each image, we infer the class for the 8 basic transformations of the 
      image; this is, besides the original image, we obtain the inference for 
      rotated 90 deg, 180 deg, and 270 deg versions, and the mirrored versions.
    - The metrics to compute are:
      * Accuracy, Precision, Recall, and F1 for the default version (sufix '_Def').
      * Accuracy, Precision, Recall, and F1 considering the 8 versions of the  
        images (sufix '_All').
    - A metric named 'agreement' is computed, which simply provides the percentage 
      of images that were classified as the same class for the 8 transformations, 
      regardless of whether the classification was correct or not.
    Other files created:
    - A confusion matrix will be created and saved in the "args.save_folder"
    - An excel file with the true class and prediction of the default images, and 
      the agreement metric.
    - The same info form the excel file is also stored in a npz file.
    - All the metrics are displayed on screen (or log file).
    """
    # Analyze the 4 types of rotation and 2 types of flipping (8 types in total).
    tst_Rot0_F0 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot1_F0 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot2_F0 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot3_F0 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot0_F1 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot1_F1 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot2_F1 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_Rot3_F1 = [np.ndarray(shape=(len(tst_img[ii]),), dtype=np.float32) for ii in range(len(args.class_folders))]
    tst_lbl = [ii*np.ones(shape=(len(tst_img[ii]),), dtype=np.uint8) for ii in range(len(args.class_folders))]

    # Compute the estimations for all 8 transformations.
    for iCls in range(len(tst_img)):
        for iTst in range(len(tst_img[iCls])):
            x_batch_R0, _ = build_validation_batch(tst_img[iCls][iTst:iTst+1], 
                                                imgLabel=iCls, args=args)
            x_batch_R1 = np.rot90(x_batch_R0, k=1, axes=(1, 2))
            x_batch_R2 = np.rot90(x_batch_R0, k=2, axes=(1, 2))
            x_batch_R3 = np.rot90(x_batch_R0, k=3, axes=(1, 2))
            tst_Rot0_F0[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R0))
            tst_Rot1_F0[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R1))
            tst_Rot2_F0[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R2)) 
            tst_Rot3_F0[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R3)) 

            x_batch_R0 = np.flip(x_batch_R0, axis=1)
            x_batch_R1 = np.flip(x_batch_R1, axis=1)
            x_batch_R2 = np.flip(x_batch_R2, axis=1)
            x_batch_R3 = np.flip(x_batch_R3, axis=1)
            tst_Rot0_F1[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R0))
            tst_Rot1_F1[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R1)) 
            tst_Rot2_F1[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R2)) 
            tst_Rot3_F1[iCls][iTst:iTst+1] = np.argmax(model.predict_on_batch(x_batch_R3)) 

    # Unravel the results in a single list
    flat_tst_lbl     = np.array([item for sublist in tst_lbl for item in sublist])
    flat_tst_Rot0_F0 = np.array([item for sublist in tst_Rot0_F0 for item in sublist])
    flat_tst_Rot1_F0 = np.array([item for sublist in tst_Rot1_F0 for item in sublist])
    flat_tst_Rot2_F0 = np.array([item for sublist in tst_Rot2_F0 for item in sublist])
    flat_tst_Rot3_F0 = np.array([item for sublist in tst_Rot3_F0 for item in sublist])
    flat_tst_Rot0_F1 = np.array([item for sublist in tst_Rot0_F1 for item in sublist])
    flat_tst_Rot1_F1 = np.array([item for sublist in tst_Rot1_F1 for item in sublist])
    flat_tst_Rot2_F1 = np.array([item for sublist in tst_Rot2_F1 for item in sublist])
    flat_tst_Rot3_F1 = np.array([item for sublist in tst_Rot3_F1 for item in sublist])

    # Compute the overall accuracy 
    tst_All = np.copy(flat_tst_Rot0_F0)
    tst_All = np.append(tst_All, flat_tst_Rot1_F0, axis=0)
    tst_All = np.append(tst_All, flat_tst_Rot2_F0, axis=0)
    tst_All = np.append(tst_All, flat_tst_Rot3_F0, axis=0)
    tst_All = np.append(tst_All, flat_tst_Rot0_F1, axis=0)
    tst_All = np.append(tst_All, flat_tst_Rot1_F1, axis=0)
    tst_All = np.append(tst_All, flat_tst_Rot2_F1, axis=0)
    tst_All = np.append(tst_All, flat_tst_Rot3_F1, axis=0)

    # Accuracy for the default transformation
    tst_Accuracy_Def = sk_metrics.accuracy_score(flat_tst_lbl, flat_tst_Rot0_F0)
    tst_Accuracy_All = sk_metrics.accuracy_score(np.tile(flat_tst_lbl,8), tst_All)

    # Precision and recall (sensitivity) for each class
    tst_Precision_Def = sk_metrics.precision_score(flat_tst_lbl, flat_tst_Rot0_F0, 
                                                labels=args.classes, average=None)
    tst_Precision_All = sk_metrics.precision_score(np.tile(flat_tst_lbl,8), tst_All, 
                                                labels=args.classes, average=None)
    tst_Recall_Def = sk_metrics.recall_score(flat_tst_lbl, flat_tst_Rot0_F0, 
                                             labels=args.classes, average=None)
    tst_Recall_All = sk_metrics.recall_score(np.tile(flat_tst_lbl,8), tst_All, 
                                             labels=args.classes, average=None)

    # F1 
    tst_F1_Def = sk_metrics.f1_score(flat_tst_lbl, flat_tst_Rot0_F0, 
                                     labels=args.classes, average=None)
    tst_F1_All = sk_metrics.f1_score(np.tile(flat_tst_lbl,8), tst_All, 
                                     labels=args.classes, average=None)

    # Compute the Agreement
    tst_4STD = np.copy(flat_tst_Rot0_F0[:,np.newaxis])
    tst_4STD = np.append(tst_4STD, flat_tst_Rot1_F0[:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, flat_tst_Rot2_F0[:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, flat_tst_Rot3_F0[:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, flat_tst_Rot0_F1[:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, flat_tst_Rot1_F1[:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, flat_tst_Rot2_F1[:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, flat_tst_Rot3_F1[:,np.newaxis], axis=1)
    tst_Agree = np.zeros(shape=(tst_4STD.shape[0],))
    for ii in range(tst_4STD.shape[0]):
        if np.max(tst_4STD[ii,:]) == np.min(tst_4STD[ii,:]):
            tst_Agree[ii] = 1
    Agree = 100*np.sum(tst_Agree)/tst_Agree.shape[0]


    # -----------------------------------------------------------------------------
    # Write an Excel file with the metrics for each test image
    flat_img_names = np.array([item for sublist in tst_img for item in sublist])
    flat_img_names = [word[len(args.data_folder):] for word in flat_img_names]
    df1 = pd.DataFrame({'Name': flat_img_names, 
                        'True Class': flat_tst_lbl, 
                        'Prediction': flat_tst_Rot0_F0,  
                        'Agreement': tst_Agree})
    fileOut = os.path.join(args.save_folder, 'Metrics.xlsx')
    df1.to_excel(fileOut, sheet_name='Metrics', index=False)

     # Save a file with the results
    fileOut = "TestSet_Variables__True_Pred_Agree.npz"
    fileOut = args.save_folder + fileOut
    np.savez(fileOut, 
             flat_img_names=flat_img_names, 
             tst_Agree=tst_Agree,
             flat_tst_lbl=flat_tst_lbl, 
             flat_tst_Rot0_F0=flat_tst_Rot0_F0)
    print("  Info saved!")


    # -----------------------------------------------------------------------------
    # Confusion Matrix
    class_descrip2 = ['Neutrophil (Segm.)', 'Neutrophil (band)', \
                      'Lymphocite (typ.)', 'Lymphocite (atyp.)', \
                      'Monocyte', 'Eosinophil', 'Basophil', 'Myeloblast', \
                      'Promyelocyte', 'Promyelocyte (bil.)', \
                      'Myelocyte', 'Metamyelocyte', 'Monoblast', \
                      'Erythroblast', 'Smudge cell']
                    
    fig, ax = plt.subplots(figsize=(18,15))
    confMat = sk_metrics.confusion_matrix(flat_tst_lbl, flat_tst_Rot0_F0, 
                                        normalize='true')
    df_cm = pd.DataFrame(confMat, index = [i for i in args.classes],
                        columns = [i for i in args.classes])
    cmap = sn.color_palette("Blues", as_cmap=True)
    sn.set(font_scale=1.5) # for label size
    sn.heatmap(df_cm, cmap=cmap, annot=True, fmt='.2f', annot_kws={"size": 15})
    plt.xlabel('Prediction',   fontsize=20)
    plt.ylabel('Ground Truth', fontsize=20)
    ax.set_yticklabels(class_descrip2, fontsize=10, ha='right', rotation=30)
    ax.set_xticklabels(class_descrip2, fontsize=10, ha='right', rotation=30)
    fileOut = args.save_folder + "ConfusionMatrix.png" 
    plt.savefig(fileOut) #save the figure in a file
    #plt.show()  
    plt.clf()
    plt.close(fig)


    # -----------------------------------------------------------------------------
    # Plot on display the losses and accuracy for the final testing
    print('-------------------------------------------------')
    print('TEST:')
    print('-------------------------------------------------')
    print('Accuracy (Def): %.4f' % tst_Accuracy_Def)
    print('Accuracy (All): %.4f' % tst_Accuracy_All)
    print('Agreement     : %.2f' % Agree)

    print('-------------------------------------------------')
    print('TEST: Precision, Recall, F1')
    print('-------------------------------------------------')

    print('Precision (Default)')
    for ii in range(tst_Precision_Def.shape[0]):
        print('  Class ' + str(ii+1).zfill(2) + ' - ' + \
            args.class_descrip[ii] + '\t %.4f' % tst_Precision_Def[ii]) 
    print('  Class All- \t\t\t\t %.4f' % np.mean(tst_Precision_Def)) 
    print('')
    print('Recall (Default)')
    for ii in range(tst_Precision_Def.shape[0]):
        print('  Class ' + str(ii+1).zfill(2) + ' - ' + \
            args.class_descrip[ii] + '\t %.4f' % tst_Recall_Def[ii]) 
    print('  Class All- \t\t\t\t %.4f' % np.mean(tst_Recall_Def)) 
    print('')
    print('F1 (Default)')
    for ii in range(tst_Precision_Def.shape[0]):
        print('  Class ' + str(ii+1).zfill(2) + ' - ' + \
            args.class_descrip[ii] + '\t %.4f' % tst_F1_Def[ii]) 
    print('  Class All- \t\t\t\t %.4f' % np.mean(tst_F1_Def)) 
    print('')
    print('Precision (All)')
    for ii in range(tst_Precision_Def.shape[0]):
        print('  Class ' + str(ii+1).zfill(2) + ' - ' + \
            args.class_descrip[ii] + '\t %.4f' % tst_Precision_All[ii]) 
    print('  Class All- \t\t\t\t %.4f' % np.mean(tst_Precision_All)) 
    print('')
    print('Recall (All)')
    for ii in range(tst_Precision_Def.shape[0]):
        print('  Class ' + str(ii+1).zfill(2) + ' - ' + \
            args.class_descrip[ii] + '\t %.4f' % tst_Recall_All[ii]) 
    print('  Class All- \t\t\t\t %.4f' % np.mean(tst_Recall_All))
    print('')
    print('F1 (All)')
    for ii in range(tst_Precision_Def.shape[0]):
        print('  Class ' + str(ii+1).zfill(2) + ' - ' + \
            args.class_descrip[ii] + '\t %.4f' % tst_F1_All[ii]) 
    print('  Class All- \t\t\t\t %.4f' % np.mean(tst_F1_All))


# -----------------------------------------------------------------------------

def test_model_leukemia(model, tst_img, tst_lbl, tst_nam, args):
    #Analyze the 4 types of rotation and 2 types of flipping (8 types in total).
    nTst = nTst = len(tst_img)
    tst_Rot0_F0 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot1_F0 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot2_F0 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot3_F0 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot0_F1 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot1_F1 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot2_F1 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Rot3_F1 = np.ndarray(shape=(nTst,2), dtype=np.float32)
    tst_Accuracy = np.ndarray(shape=(2,), dtype=np.float32)

    print('The number of Test images is: ', nTst)
    for iTst in range(nTst):
        x_batch_R0, _ = build_validation_batch(tst_img[iTst:iTst+1], imgLabel=1, args=args)
        x_batch_R1 = np.rot90(x_batch_R0, k=1, axes=(1, 2))
        x_batch_R2 = np.rot90(x_batch_R0, k=2, axes=(1, 2))
        x_batch_R3 = np.rot90(x_batch_R0, k=3, axes=(1, 2))
        tst_Rot0_F0[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R0) 
        tst_Rot1_F0[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R1) 
        tst_Rot2_F0[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R2) 
        tst_Rot3_F0[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R3) 

        x_batch_R0 = np.flip(x_batch_R0, axis=1)
        x_batch_R1 = np.flip(x_batch_R1, axis=1)
        x_batch_R2 = np.flip(x_batch_R2, axis=1)
        x_batch_R3 = np.flip(x_batch_R3, axis=1)
        tst_Rot0_F1[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R0)
        tst_Rot1_F1[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R1) 
        tst_Rot2_F1[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R2) 
        tst_Rot3_F1[iTst:iTst+1,:] = model.predict_on_batch(x_batch_R3) 


    # Accuracy for the 8 transformations
    tst_Rot0_F0 = tst_Rot0_F0[:,1]
    tst_Accuracy[0] = sk_metrics.accuracy_score(tst_lbl, np.round(tst_Rot0_F0))

    # Compute the overall accuracy 
    tst_All = np.copy(tst_Rot0_F0)
    tst_All = np.append(tst_All, tst_Rot1_F0[:,1], axis=0)
    tst_All = np.append(tst_All, tst_Rot2_F0[:,1], axis=0)
    tst_All = np.append(tst_All, tst_Rot3_F0[:,1], axis=0)
    tst_All = np.append(tst_All, tst_Rot0_F1[:,1], axis=0)
    tst_All = np.append(tst_All, tst_Rot1_F1[:,1], axis=0)
    tst_All = np.append(tst_All, tst_Rot2_F1[:,1], axis=0)
    tst_All = np.append(tst_All, tst_Rot3_F1[:,1], axis=0)
    tst_Accuracy[1] = sk_metrics.accuracy_score(np.tile(tst_lbl,8), np.round(tst_All))

    # Compute the Agreement
    tst_4STD = np.copy(tst_Rot0_F0[:,np.newaxis])
    tst_4STD = np.append(tst_4STD, tst_Rot1_F0[:,1][:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, tst_Rot2_F0[:,1][:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, tst_Rot3_F0[:,1][:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, tst_Rot0_F1[:,1][:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, tst_Rot1_F1[:,1][:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, tst_Rot2_F1[:,1][:,np.newaxis], axis=1)
    tst_4STD = np.append(tst_4STD, tst_Rot3_F1[:,1][:,np.newaxis], axis=1)
    tst_Agree = np.sum(np.round(tst_4STD), axis=1)/tst_4STD.shape[1]
    tst_Agree[tst_Agree < 0.5] = 1 - tst_Agree[tst_Agree < 0.5]

    # AUC and 
    tst_AUC_def = sk_metrics.roc_auc_score(tst_lbl, tst_Rot0_F0)
    tst_AUC_all = sk_metrics.roc_auc_score(np.tile(tst_lbl,8), tst_All)

    # F1 weighted, Precision and recall (sensitivity) 
    tst_F1_Def = sk_metrics.f1_score(tst_lbl, np.round(tst_Rot0_F0), 
                                    labels=args.classes, average='weighted')
    tst_F1_All = sk_metrics.f1_score(np.tile(tst_lbl,8), np.round(tst_All), 
                                    labels=args.classes, average='weighted')

    tst_Precis_Def = sk_metrics.precision_score(tst_lbl, np.round(tst_Rot0_F0), 
                                    labels=args.classes, average='weighted')
    tst_Precis_All = sk_metrics.precision_score(np.tile(tst_lbl,8), np.round(tst_All), 
                                    labels=args.classes, average='weighted')

    tst_Recall_Def = sk_metrics.recall_score(tst_lbl, np.round(tst_Rot0_F0), 
                                    labels=args.classes, average='weighted')
    tst_Recall_All = sk_metrics.recall_score(np.tile(tst_lbl,8), np.round(tst_All), 
                                    labels=args.classes, average='weighted')


    # -----------------------------------------------------------------------------
    # Write a Excel file with the metrics for each test image
    df1 = pd.DataFrame({'Name': tst_nam, 
                        'True Class': tst_lbl, 
                        'Prediction': tst_Rot0_F0,  
                        'Agreement': tst_Agree})
    fileOut = os.path.join(args.save_folder, 'Metrics.xlsx')
    df1.to_excel(fileOut, sheet_name='Metrics_TestSet', index=False)

    fileOut = "TestSet_Variables__True_Pred_Agree.npz"
    fileOut = args.save_folder + fileOut
    np.savez(fileOut, tst_nam=tst_nam, tst_Agree=tst_Agree,
                      tst_lbl=tst_lbl, tst_Rot0_F0=tst_Rot0_F0)
    print("  Info saved!")


    # -----------------------------------------------------------------------------
    # Plot the losses and accuracy for the final testing
    print('-------------------------------------------------')
    print('TEST:')
    print('-------------------------------------------------')
    print('Accuracy (Def): \t %.6f' % tst_Accuracy[0])   
    print('Accuracy (All): \t %.6f' % tst_Accuracy[1])   
    print('Agreement:      \t %.4f' % np.mean(tst_Agree))  

    print('-------------------------------------------------')
    print('TEST: AUC and F1')
    print('-------------------------------------------------')
    print('AUC (Def): \t %.6f' % tst_AUC_def)  
    print('AUC (All): \t %.6f' % tst_AUC_all)  

    print('-------------------------------------------------')
    print('TEST: F1, Precision, Recall (weighted)')
    print('-------------------------------------------------')

    print('Precision (Default): \t %.6f' % tst_Precis_Def)
    print('Recall (Default):    \t %.6f' % tst_Recall_Def) 
    print('F1 (Default):        \t %.6f' % tst_F1_Def)
    print('')
    print('Precision (All): \t %.6f' % tst_Precis_All)
    print('Recall (All):    \t %.6f' % tst_Recall_All) 
    print('F1 (All):        \t %.6f' % tst_F1_All)

