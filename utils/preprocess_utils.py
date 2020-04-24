import numpy as np
import cv2
import random
import os


# def show(img, title='无标题'):
#     """
#     本地测试时展示图片
#     :param img:
#     :param name:
#     :return:
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.font_manager import FontProperties
#     font = FontProperties(fname='/Users/yanmeima/workspace/ocr/crnn/data/data_generator/fonts/simhei.ttf')
#     plt.title(title, fontsize='large', fontweight='bold', FontProperties=font)
#     plt.imshow(img)
#     plt.show()


def get_patches(img):
    dim = 256
    h, w = img.shape[:2]

    # 如果图像宽高小于256，补齐到256
    if h < dim:
        # copyMakeBorder(img,top,bottom,left,right,borderType,colar
        # 把图像边界补上白色
        img = cv2.copyMakeBorder(img, 0, dim - h, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    if w < dim:
        img = cv2.copyMakeBorder(img, 0, 0, 0, dim - w, cv2.BORDER_CONSTANT, value=(255,255,255))

    # 转成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    #看是256的几倍
    hNum, wNum = int(h / dim), int(w / dim)

    hPatchStart = 0  # if hNum < 4 else 1
    wPatchStart = 0  # if wNum < 4 else 1
    hPatchEnd = hNum  # if hNum < 4 else hNum - 1
    wPatchEnd = wNum  # if wNum < 4 else wNum - 1
    candiIdx = []
    backupIdx = []
    # 按照256x256去裁图像
    candidate_patches = []
    for hIdx in range(hPatchStart, hPatchEnd):
        for wIdx in range(wPatchStart, wPatchEnd):
            hStart = hIdx * dim
            wStart = wIdx * dim
            backupIdx.append((hStart, wStart))
            grayCrop = gray[hStart:(hStart + dim), wStart:(wStart + dim)]
            candidate_patches.append(grayCrop)

    # 随机取32个，之前按顺序来，并且只有16个，对比较大的图，如3000x4000这类的，就会值切到某个边缘，如果边缘没有单据图像就惨了
    # 所以，现在是随机取32个。改进后，效果好一些了。
    patch_idxes = np.arange(0, len(candidate_patches))
    #print("patch_idxes:",patch_idxes)
    random.shuffle(patch_idxes)

    # 从随机里面只取32个出来，32 hardcode了
    # TODO 还需要做优化
    done_counter = 0
    for patch_idx in patch_idxes:
        # 用MSER+NMS，找有多少个包含文字的框
        candidate_patch = candidate_patches[patch_idx]
        boxCnt = getTextBoxCnt(candidate_patch)
        #show(candidate_patch,str(boxCnt))
        #print(hIdx, wIdx, boxCnt)
        # >5个才作为备选，用于检验歪斜
        if boxCnt >= 5:
            candiIdx.append(backupIdx[patch_idx])
            done_counter+=1
        if done_counter>=32: break

    if len(candiIdx) == 0:candiIdx = backupIdx

    # 做一下标准化
    patches = []
    for hStart, wStart in candiIdx[:32]:
        patch = img[hStart:(hStart + dim), wStart:(wStart + dim)]
        #因为后面有强制压缩成224 * 224和标准化，所以这里不需要标准化
        patch = (patch - patch.mean()) / patch.std() # 做一下标准化----------！！！因为后面进VGG之前要减均值，所以这里我没有做标准化
        patches.append(patch)
    #patches = np.stack(patches, axis=0)
    return patches

# 入参是256x256的小图，一堆
# 返回是，经过MSER后剩余的框的个数
def getTextBoxCnt(gray):
    # 使用最大稳定极值区域MSER找所有的明显的文字区域
    mser = cv2.MSER_create(_min_area=20, _max_area=600)
    regions, boxes = mser.detectRegions(gray)
    # print(regions[:5])
    # print(boxes[:5])
    # print(type(boxes), len(boxes), boxes.shape)
    keep = []
    for box in boxes:
        x, y, w, h = box
        keep.append([x, y, x + w, y + h])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # 对IOU>0.3的进行NMS筛选，剩下的是靠谱的框（在remains里面）
    remains = nms(np.array(keep), 0.3)
    # print(remains.shape)
    # for x1, y1, x2, y2 in remains:
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return remains.shape[0]

# nms实现
def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0: return np.array([])

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs,
                         np.concatenate(([last],
                                        np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


if __name__ == '__main__':
    img = cv2.imread("data/debug/images/ocr_o_RJifyfKN1569570661198_e4UaHf891569570683130_7250508932517050569.JPG")
    #print(img.shape)
    patches = get_patches(img)
    i = 0
    for p in patches:
        cv2.imwrite(os.path.join("data/debug/patches/" + str(i) + ".jpg"),p)
        i +=1

    #print(patches)