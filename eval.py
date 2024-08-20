import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
from hausdorff import hausdorff_distance
from skimage.measure import label
# import numpy as np
#
# 创建示例的标签数组和预测数组
# labels = np.array([[1, 1, 0, 0],
#                    [1, 0, 0, 1],
#                    [0, 1, 1, 1]])
#
# y_pred = np.array([[1, 1, 0, 0],
#                    [1, 0, 1, 1],
#                    [0, 1, 0, 1]])
#
# 计算交集
# intersection = np.logical_and(labels, y_pred)#通过使用np.logical_and()函数，我们可以逐元素比较labels和y_pred
#并获得一个布尔数组，其中对应位置为True表示两个数组在该位置上都为True，
# intersection_count = np.sum(intersection)
#
# 计算并集
# union = np.logical_or(labels, y_pred)
# union_count = np.sum(union)
#
# 计算交并比
# iou = intersection_count / union_count
#
# print("Intersection:", intersection_count)
# print("Union:", union_count)
# print("Intersection over Union (IoU):", iou)

# def iou_metric(y_true_in, y_pred_in, print_table=False):
#     labels, _ = torch.unique(torch.tensor(y_true_in > 0.5).float(), return_inverse=True)
#     y_pred, _ = torch.unique(torch.tensor(y_pred_in > 0.5).float(), return_inverse=True)
#     #https://blog.csdn.net/t20134297/article/details/108235355
#
#     max_value = torch.max(labels.max(), y_pred.max())
#     hist_bins = int(max_value) + 1
#
#     hist_labels = torch.bincount(labels.to(torch.int64), minlength=hist_bins)
#     hist_y_pred = torch.bincount(y_pred.to(torch.int64), minlength=hist_bins)
#
#     intersection = torch.bincount(torch.cat((labels.flatten(), y_pred.flatten())).to(torch.int64), minlength=hist_bins)
#
#     area_true = hist_labels.unsqueeze(-1)
#     area_pred = hist_y_pred.unsqueeze(0)
#
#     union = area_true + area_pred - intersection
#
#     intersection = intersection[1:]
#     union = union[1:]
#     union[union == 0] = 1e-9
#
#     iou = intersection / union
#
#     def precision_at(threshold, iou):
#         matches = iou > threshold
#         true_positives = torch.sum(matches, dim=1) == 1
#         false_positives = torch.sum(matches, dim=0) == 0
#         false_negatives = torch.sum(matches, dim=1) == 0
#         tp, fp, fn = torch.sum(true_positives), torch.sum(false_positives), torch.sum(false_negatives)
#         return tp, fp, fn
#
#     prec = []
#     if print_table:
#         print("Thresh\tTP\tFP\tFN\tPrec.")
#     for t in torch.arange(0.5, 1.0, 0.05):
#         tp, fp, fn = precision_at(t, iou)
#         if (tp + fp + fn) > 0:
#             p = tp / (tp + fp + fn)
#         else:
#             p = 0
#         if print_table:
#             print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t.item(), tp.item(), fp.item(), fn.item(), p.item()))
#         prec.append(p)
#
#     if print_table:
#         print("AP\t-\t-\t-\t{:1.3f}".format(torch.mean(torch.tensor(prec)).item()))
#     return torch.mean(torch.tensor(prec, dtype=torch.float32)).item


def iou_metric(y_true_in, y_pred_in, print_table=False):
    # labels = label(y_true_in > 0.5)  # label用于将二标签转换为具体的标签名称
    # y_pred = label(y_pred_in > 0.5)  # 预测值大于0.5的标记为1 小于则标记为0
    y_true_in = y_true_in.cpu().numpy()
    y_pred_in = y_pred_in.cpu().numpy()
    # print(y_pred_in.type())
    # print(y_true_in.type())
    # print("y_true", y_true_in.shape)
    # print(y_true_in)
    # print("y_pred", y_pred_in.shape)
    # print(y_pred_in)



    # 计算出真实类别数和预测的聚类数
    true_objects = len(np.unique(y_true_in))  # np.unique()函数取出重复元素并将元素从小到达排列，从而返回一个无重复元素的列表或者元组
    pred_objects = len(np.unique(y_pred_in))  # pre_objects中不同类别的数量（也就是有几类）

    # print("true_objects", true_objects)
    # print("pred_objects", pred_objects)
    # exit(0)

    # 计算的是y_true_in和y_pred_in的交集
    intersection = np.histogram2d(y_true_in.flatten(), y_pred_in.flatten(), bins=(true_objects, pred_objects))[0]
    # bins是一个二元组，表示在两个维度上的箱子数目（箱子指的是将数据范围划分成一系列等宽的区间，每个箱子代表一个区间范围，用于统计数据落在该范围内的样本数量）
    # 这里两个维度上的数据范围参数range默认值是none  它会根据数据中的最小值和最大值作为取值范围，使得直方图能够覆盖整个数据范围
    # np.histogram2d的第一个返回值是一个二维数组（x_bins,y_bins）,及x轴和y轴上的箱子数目，数组中每个元素代表了在对应箱子内的样本数量

    # y_true_in.flatten()是一个numpy方法 用于将多维数组转化为一维数组

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true_in, bins=true_objects)[0]  # 生成一维直方图 bins表示区间的个数  最后返回每个区间中元素的个数
    area_pred = np.histogram(y_pred_in, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)  # 在最后一个维度上扩展一个维度
    area_pred = np.expand_dims(area_pred, 0)  # 在第一个维度上扩展一个维度

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]  # 去掉第一行第一列 因为第一行第一列是背景类别 不参与运算 这部分与真实对象和预测对象的交集无关
    union = union[1:, 1:]
    union[union == 0] = 1e-9  # 避免出现除以0的错误

    # Compute the intersection over union计算交并比
    iou = intersection / union

    # print('IOU {}'.format(iou))
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

# def iou_metric(gt, pred, print_table=False):
#     # print(gt.type)
#     # print("gt----", gt)
#     # print("###########")
#     # print(pred.type)
#     # print("pred----", pred)
#     intersection = torch.logical_and(gt, pred)  # 计算相交部分
#     union = torch.logical_or(gt, pred)  # 计算并集
#     iou = torch.sum(intersection).item() / torch.sum(union).item()
#     return iou

def mean_iou(predicted, target, num_classes):
    iou_sum = 0.0
    for cls in range(num_classes):
        intersection = torch.logical_and(predicted == cls, target == cls).sum().item()
        union = torch.logical_or(predicted == cls, target == cls).sum().item()
        iou = intersection / (union + 1e-8)
        iou_sum += iou
    mean_iou = iou_sum / num_classes
    return mean_iou

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.view(-1)#将张量展平为一维
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)#交集近似于预测图和真实图之间的点乘，再将点乘元素结果相加
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    #计算并集，采用元素相加  分子乘以2是因为分母存在重复计算pred和true之间共同元素的原因
    #这里的smooth通常是一个很小的正数，当分母中的真实值和预测值之和接近零时，除法操作可能会导致除以0的错误
    score = score.item()
    return score


def hausdorff_distance_batch(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return hausdorff_distance(y_true, y_pred)
    else:
        batch_size = y_true.shape[0]
        hd = 0.
    for batch in range(batch_size):
        y_true_np = y_true[batch].cpu().numpy()
        y_pred_np = y_pred[batch].cpu().numpy()
        hd += hausdorff_distance(y_true_np, y_pred_np)
    return hd / batch_size


def iou_metric_batch(y_true_in, y_pred_in):#每一批样本的平均iou
    batch_size = y_true_in.shape[0]
    value = 0.
    for batch in range(batch_size):
        value = value + iou_metric(y_true_in[batch], y_pred_in[batch])
    return value/batch_size


def evalResult(gt, pred, target_size=(256, 256), num_classes=2):
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)#将numpy转换成pytorch张量对象

    gt = torch.squeeze(gt)#从张量中移除尺寸为1的维度
    pred = torch.squeeze(pred)

    gt = gt.cpu()
    pred = pred.cpu()

    r_acc = accuracy_score(gt.flatten(), pred.flatten())#计算acc

    r_pr = precision_score(gt.flatten(), pred.flatten(), average='micro')#计算precision

    r_rc = recall_score(gt.flatten(), pred.flatten(), average='micro')#就算recall

   
    MeanIoU = mean_iou(pred, gt, 2)#计算的是单个样本的平均iou  在每一类上的

    dc = dice_coeff(gt, pred)#dice  计算真实值与预测值之间的相似性  越高越好

    hd = hausdorff_distance_batch(gt, pred)#表示真实值与预测值之间的差异  越小越好

    Mean_batch_IoU = iou_metric_batch(gt, pred)#平均iou 图像分割等任务中预测结果和真实目标之间的重叠程度 越大越好  每一批的样本的平均iou

    r_mae = mean_absolute_error(gt.flatten().numpy(), pred.flatten().numpy())#计算平均绝对误差 MAE 它对预测误差的绝对值进行求和取平均  故计算出的值越小表示预测结果与真实值越接近


    # print("Accuracy=", r_acc, "Precision=", r_pr, "Recall=", r_rc, "MeanIoU=",MeanIoU, "DiceCoefficient=", dc, "HD=", hd,
    #       "Mean_batch_IoU=", Mean_batch_IoU, "MAE=", r_mae)
    return r_acc, r_pr, r_rc, MeanIoU, dc, hd, Mean_batch_IoU, r_mae

