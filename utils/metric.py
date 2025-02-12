import os, sys
import numpy as np

from multiprocessing import Pool

# import copy_reg
import pickle
import types


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


# pickle.dump(types.MethodType, _pickle_method)


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(
        (
            "background",  # always index 0
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
    )

    for i, iou in enumerate(j_list):
        print("class {:2d} {:12} IU {:.2f}".format(i, classes[i], j_list[i]))

    print("meanIOU: " + str(aveJ) + "\n")
    if save_path:
        with open(save_path, "w") as f:
            for i, iou in enumerate(j_list):
                f.write(
                    "class {:2d} {:12} IU {:.2f}".format(i, classes[i], j_list[i])
                    + "\n"
                )
            f.write("meanIOU: " + str(aveJ) + "\n")
    return aveJ


class ConfusionMatrix(object):
    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert np.max(pred) <= self.nclass
        assert len(gt) == len(pred)
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert matrix.shape == self.M.shape
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_sum = []
        jaccard_perclass = []
        for i in range(self.nclass):
            jaccard_perclass.append(
                self.M[i, i]
                / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i])
            )
            if not self.M[i, i] == 0:
                jaccard_sum.append(
                    self.M[i, i]
                    / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i])
                )

        return np.sum(jaccard_sum) / self.nclass, jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert len(gt) == len(pred)
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


if __name__ == "__main__":
    args = parse_args()

    m_list = []
    data_list = []
    test_ids = [i.strip() for i in open(args.test_ids) if not i.strip() == ""]
    for index, img_id in enumerate(test_ids):
        if index % 100 == 0:
            print("%d processd" % (index))
        pred_img_path = os.path.join(args.pred_dir, img_id + ".png")
        gt_img_path = os.path.join(args.gt_dir, img_id + ".png")
        pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        # show_all(gt, pred)
        data_list.append([gt.flatten(), pred.flatten()])

    ConfM = ConfusionMatrix(args.class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    with open(args.save_path, "w") as f:
        f.write("meanIOU: " + str(aveJ) + "\n")
        f.write(str(j_list) + "\n")
        f.write(str(M) + "\n")
