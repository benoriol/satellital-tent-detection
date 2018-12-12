import numpy as np
import cv2
from PIL import Image

def BinaryAccuracyError(prediction, gt):

    assert prediction.shape == gt.shape




    error = np.absolute(prediction - gt)
    error = np.sum(error)
    error = error/(gt.shape[0] * gt.shape[1])

    prediction = cv2.resize(prediction, (700, 700))
    gt = cv2.resize(gt, (700, 700))

    cv2.imshow('image', gt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image', prediction)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return error

def getfolder(file):

    file = file.split('/')
    file = file[:-1]

    if len(file) > 0:
        return '/'.join(file)
    else:
        return '.'

def GetErr(PREDICTION_METADATA, GT_METADATA, PRED_FOLDER, GT_FOLDER):

    pred_len = 0
    gt_len = 0

    for x in open(PREDICTION_METADATA):
        pred_len += 1

    for x in open(GT_METADATA):
        gt_len += 1

    if pred_len != gt_len:
        raise ValueError('The two metadata files must have the same length')

    acc_err = 0

    worse_err = -1

    for i, gt_path, pred_path in zip(range(gt_len), open(GT_METADATA), open(PREDICTION_METADATA)):

        pred_path = pred_path.strip()
        gt_path = gt_path.strip()

        # gt = cv2.imread(GT_FOLDER + '/' + gt_path, 0)
        gt = Image.open(GT_FOLDER + '/' + gt_path)
        gt = np.array(gt, dtype='float')
        # print(gt.shape)
        # quit()
        pred = cv2.imread(PRED_FOLDER + '/' + pred_path, 0) / 255

        print(gt_path + '\tvs \t' + pred_path)

        pred = cv2.resize(pred, gt.shape)

        err = BinaryAccuracyError(pred, gt)

        acc_err += err

        if err > worse_err:
            worse_err = err
            worse_case = pred_path

    acc_err /= pred_len

    return acc_err, worse_err, worse_case



if __name__ == '__main__':
    PREDICTION_METADATA = 'results/VILLAGES1/results_metadata.txt'
    GT_METADATA = 'results/VILLAGES1/mask_metadata.txt'

    PRED_FOLDER = getfolder(PREDICTION_METADATA)
    GT_FOLDER = getfolder(GT_METADATA)

    acc_err, worse_err, worse_case = GetErr(PREDICTION_METADATA, GT_METADATA,
                                            PRED_FOLDER, GT_FOLDER)

    print("Accuracy: " + str(acc_err))
    print("Worse case: " + str(worse_err) + "   at image: " + str(worse_case))


