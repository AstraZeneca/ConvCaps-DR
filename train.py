
import numpy as np
import datetime
from plot_metrics import plot_metrics_TrnVal
import tensorflow.keras.backend as K
from build_batches import build_training_batch_myeloid, build_validation_batch
from build_batches import build_training_batch_leukemia
from plot_metrics import plot_metrics_TrnVal, print_metrics_model, print_time_cost


def train_model_myeloid(model, trn_img, val_img, args):
    # Initialize some variables to track the loss and the AUC
    nMtrc = 2         # The different metrics to record
    trn_met = np.ndarray(shape=(nMtrc,0), dtype=np.float32)
    val_met = np.ndarray(shape=(nMtrc,0), dtype=np.float32)
    val_metNoW = np.ndarray(shape=(nMtrc,0), dtype=np.float32) #Non-Weighted
    tTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32) # Temporal variable

    # Variables to keep track of time expenditure
    timeIni = datetime.datetime.now().replace(microsecond=0)   # Initial time
    timeCur = datetime.datetime.now()  # Current time (update in each iteration)        

    # -------------------------------------------------------------------------
    # If continuing training, load the weights and variables. 
    if args.current_epoch > 0:
        model.load_weights(args.save_folder + args.weights)
        data = np.load(args.save_folder + 'epoch' + str(args.current_epoch) + 
                       '_variables.npz', allow_pickle=True)
        trn_img = data['trn_img']
        val_img = data['val_img']
        trn_met = data['trn_met']
        val_met = data['val_met']
        val_metNoW = data['val_metNoW']

        # Plot the losses and accuracy for training/validation
        x1 = 1 + np.arange(trn_met.shape[1])
        plot_metrics_TrnVal(x1, trn_met[0], val_met[0], val_metNoW[0],
                            title='Loss', check_best='min', 
                            legend=['Training', 'Validation', 'Validation NoW'],
                            save_folder=args.save_folder)
        plot_metrics_TrnVal(x1, trn_met[1], val_met[1], val_metNoW[1], 
                            title='Accuracy', ylabel='Batch accuracy', 
                            ylim=[0,1], check_best='max', 
                            legend=['Training', 'Validation', 'Validation NoW'],
                            save_folder=args.save_folder)
    # -------------------------------------------------------------------------

    for iEpoch in range(args.current_epoch, args.epochs):
        # Change learning rate
        lr_new = args.lr_init*(args.lr_decay ** iEpoch)
        K.set_value(model.optimizer.lr, lr_new)
        print('New learning rate: %.6f' % model.optimizer.lr)

        for iIter in range(args.iterations): 
            # -----------------------------------------------------------------
            # TRAINING. Prepare one batch and feed it to the CNN
            batch_data, batch_labl, _ = build_training_batch_myeloid(trn_img, 
                                                                     args=args)
            tLoss = model.train_on_batch(batch_data, batch_labl) #sample_weight=batch_wght)
            tTemp = np.append(tTemp, np.array(tLoss)[:,np.newaxis], axis=1)
            
            # -----------------------------------------------------------------
            # After training iter2eval batches, average the losses 
            if ((iIter+1) % args.iter2eval == 0): 
                # Save the train metrics, plot them, and reset metrics
                trn_met = np.append(trn_met, np.mean(tTemp,axis=1)[:,np.newaxis], 
                                    axis=1)
                tTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32) # Reset
                print_time_cost(timeCur, eCurr=iEpoch+1, strT='Training')
                print_metrics_model(trn_met, iIter+1, typeSet='Training')                

                # -------------------------------------------------------------
                # VALIDATION. Save the non-weighted (NoW) and weighted metrics
                vTempNoW = np.ndarray(shape=(nMtrc,0), dtype=np.float32) 
                vTempCls = np.ndarray(shape=(nMtrc,0), dtype=np.float32) 
                for iCls in range(len(val_img)):
                    vTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32) 
                    for iVal in range(len(val_img[iCls])):
                        batch_data, batch_labl = build_validation_batch(
                            val_img[iCls][iVal:iVal+1], 
                            imgLabel=iCls, args=args)
                        vLoss = model.test_on_batch(batch_data, batch_labl) 
                        vTemp = np.append(vTemp, np.array(vLoss)[:,np.newaxis], 
                                        axis=1)

                    # Compure the averaged metrics for one class and append them
                    vTempCls = np.append(vTempCls, 
                                         np.mean(vTemp,axis=1)[:,np.newaxis], 
                                         axis=1)
                    vTempNoW = np.append(vTempNoW, vTemp, axis=1)
            
                # Compute the weighted metrics, save them, and plot them
                val_met = np.append(val_met, 
                    np.mean(vTempCls,axis=1)[:,np.newaxis], axis=1)
                val_metNoW = np.append(val_metNoW, 
                     np.mean(vTempNoW,axis=1)[:,np.newaxis], axis=1)
                print_metrics_model(val_met, iIter+1, 
                                    typeSet='Validation Negatives')
                print_metrics_model(val_metNoW, iIter+1, 
                                    typeSet='Validation Negatives - NonWeighted')

                # Plot the time again, including now the validation
                print_time_cost(timeCur, timeIni, iCurr=iIter+1, 
                                iNmbr=args.iterations, eCurr=iEpoch+1, 
                                eNmbr=args.epochs, it2ev=args.iter2eval, 
                                flagFuture=True, strT='Training + Validation')
                timeCur = datetime.datetime.now()  # Reset timestamp 

        # ---------------------------------------------------------------------
        # End of an epoch 
        if args.save_model:
            fileOut = ("epoch" + str(1+iEpoch) + "_weights.h5")
            fileOut = args.save_folder + fileOut
            model.save_weights(fileOut, overwrite=True)
            print("  Model saved!")
        

        # Save some variables for further analysis later
        if args.save_variables:
            fileOut = ("epoch" + str(1+iEpoch) + "_variables.npz")
            fileOut = args.save_folder + fileOut
            np.savez(fileOut, 
                     trn_met=trn_met, trn_img=trn_img, 
                     val_met=val_met, val_img=val_img, val_metNoW=val_metNoW)
            print("  Variables saved!")

        
        # ---------------------------------------------------------------------
        # Plot the losses and accuracy for training/validation
        x1 = 1 + np.arange(trn_met.shape[1])
        plot_metrics_TrnVal(x1, trn_met[0], val_met[0], val_metNoW[0],
                            title='Loss', check_best='min', 
                            legend=['Training', 'Validation', 'Validation NoW'],
                            save_folder=args.save_folder)
        plot_metrics_TrnVal(x1, trn_met[1], val_met[1], val_metNoW[1], 
                            title='Accuracy', ylabel='Batch accuracy', 
                            check_best='max', ylim=[0,1],
                            legend=['Training', 'Validation', 'Validation NoW'],
                            save_folder=args.save_folder)



def train_model_leukemia(model, trnPos_img, trnNeg_img, valPos_img, valNeg_img, 
                         args):
    # Initialize some variables to track the loss and the AUC
    nMtrc = 2         # The different metrics to record
    trnAll_met = np.ndarray(shape=(nMtrc,0), dtype=np.float32)
    valPos_met = np.ndarray(shape=(nMtrc,0), dtype=np.float32)
    valNeg_met = np.ndarray(shape=(nMtrc,0), dtype=np.float32)
    tTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32)
    vTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32)

    # Variables to keep track of time expenditure
    timeIni = datetime.datetime.now().replace(microsecond=0)   # Initial time
    timeCur = datetime.datetime.now()  # Current time (update in each iteration)        

    # -------------------------------------------------------------------------
    # If continuing training, load the weights and variables. 
    if args.current_epoch > 0:
        model.load_weights(args.save_folder + args.weights)
        data = np.load(args.save_folder + 'epoch' + str(args.current_epoch) + 
                       '_variables.npz', allow_pickle=True)
        trnPos_img = data['trnPos_img']
        trnNeg_img = data['trnNeg_img']
        trnAll_met = data['trnAll_met']
        valPos_img = data['valPos_img']
        valPos_met = data['valPos_met']
        valNeg_img = data['valNeg_img']
        valNeg_met = data['valNeg_met']  

        # Plot the losses and accuracy for training/validation
        x1 = 1 + np.arange(trnAll_met.shape[1])
        plot_metrics_TrnVal(x1, trnAll_met[0], valPos_met[0], valNeg_met[0],
                        title='Loss', check_best='min', 
                        legend=['Training', 'Validation Pos', 'Validation Neg'],
                        save_folder=args.save_folder)
        plot_metrics_TrnVal(x1, trnAll_met[1], valPos_met[1], valNeg_met[1], 
                        title='Accuracy', ylabel='Batch accuracy', 
                        ylim=[0.5,1], check_best='max', 
                        legend=['Training', 'Validation Pos', 'Validation Neg'],
                        save_folder=args.save_folder)
    
    # -------------------------------------------------------------------------
    nPosVal = valPos_img.shape[0]
    nNegVal = valNeg_img.shape[0]
    
    for iEpoch in range(args.current_epoch, args.epochs):
        # Change learning rate
        lr_new = args.lr_init*(args.lr_decay ** iEpoch)
        K.set_value(model.optimizer.lr, lr_new)
        print('New learning rate: %.6f' % model.optimizer.lr)

        for iIter in range(args.iterations): 
            # -----------------------------------------------------------------
            # TRAINING. Prepare one batch and feed it to the CNN
            batch_data, batch_labl, _ = build_training_batch_leukemia(
                trnPos_img, trnNeg_img, args=args)
            tLoss = model.train_on_batch(batch_data, batch_labl) #sample_weight=batch_wght)
            tTemp = np.append(tTemp, np.array(tLoss)[:,np.newaxis], axis=1)
            
            # -----------------------------------------------------------------
            # After training iter2eval batches, average the losses 
            if ((iIter+1) % args.iter2eval == 0): 
                # Save the train metrics, plot them, and reset metrics
                trnAll_met = np.append(trnAll_met, 
                                np.mean(tTemp,axis=1)[:,np.newaxis], axis=1)
                tTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32) # Reset
                print_time_cost(timeCur, eCurr=iEpoch+1, strT='Training')
                print_metrics_model(trnAll_met, iIter+1, typeSet='Training')                    

                # -------------------------------------------------------------
                # VALIDATION. Positive images
                for iVal in range(nPosVal):
                    batch_data, batch_labl = build_validation_batch(
                        valPos_img[iVal:iVal+1], imgLabel=1, args=args)
                    vLoss = model.test_on_batch(batch_data, batch_labl) 
                    vTemp = np.append(vTemp, np.array(vLoss)[:,np.newaxis], 
                                      axis=1)
                
                # Save the metrics for the Positive images, plot them, and reset 
                valPos_met = np.append(valPos_met, 
                                np.mean(vTemp,axis=1)[:,np.newaxis], axis=1)
                vTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32) # Reset
                print_metrics_model(valPos_met, iIter+1, 
                                    typeSet='Validation Positives')  

                # -------------------------------------------------------------
                # VALIDATION.  Negative images.
                for iVal in range(nNegVal):
                    batch_data, batch_labl = build_validation_batch(
                        valNeg_img[iVal:iVal+1], imgLabel=0, args=args)
                    vLoss = model.test_on_batch(batch_data, batch_labl) 
                    vTemp = np.append(vTemp, np.array(vLoss)[:,np.newaxis], 
                                      axis=1)
                
                # Save the metrics for Negative images, plot them, and reset 
                valNeg_met = np.append(valNeg_met, 
                                np.mean(vTemp,axis=1)[:,np.newaxis], axis=1)
                vTemp = np.ndarray(shape=(nMtrc,0), dtype=np.float32) # Reset
                print_metrics_model(valNeg_met, iIter+1, 
                                    typeSet='Validation Negatives')

                # Plot the time again, including now the validation
                print_time_cost(timeCur, timeIni, iCurr=iIter+1, 
                                iNmbr=args.iterations, eCurr=iEpoch+1, 
                                eNmbr=args.epochs, it2ev=args.iter2eval, 
                                flagFuture=True, strT='Training + Validation')
                timeCur = datetime.datetime.now()  # Reset timestamp 

        # ---------------------------------------------------------------------
        # End of an epoch 
        if args.save_model:
            fileOut = ("epoch" + str(1+iEpoch) + "_weights.h5")
            fileOut = args.save_folder + fileOut
            model.save_weights(fileOut, overwrite=True)
            print("  Model saved!")
        

        # Save some variables for further analysis later
        if args.save_variables:
            fileOut = ("epoch" + str(1+iEpoch) + "_variables.npz")
            fileOut = args.save_folder + fileOut
            np.savez(fileOut, trnAll_met=trnAll_met,  
                              valPos_met=valPos_met, valNeg_met=valNeg_met,
                              trnPos_img=trnPos_img, trnNeg_img=trnNeg_img, 
                              valPos_img=valPos_img, valNeg_img=valNeg_img)
            print("  Variables saved!")

        
        # ---------------------------------------------------------------------
        # Plot the losses and accuracy for training/validation
        x1 = 1 + np.arange(trnAll_met.shape[1])
        plot_metrics_TrnVal(x1, trnAll_met[0], valPos_met[0], valNeg_met[0],
                        title='Loss', check_best='min', 
                        legend=['Training', 'Validation Pos', 'Validation Neg'],
                        save_folder=args.save_folder)
        plot_metrics_TrnVal(x1, trnAll_met[1], valPos_met[1], valNeg_met[1], 
                        title='Accuracy', ylabel='Batch accuracy', 
                        ylim=[0.5,1], check_best='max', 
                        legend=['Training', 'Validation Pos', 'Validation Neg'],
                        save_folder=args.save_folder)