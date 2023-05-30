import cv2
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 0.000001)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def pre_yolo(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    # padded_img = np.ascontiguousarray(padded_img, dtype=np.float16)
    # padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class onnxModel:
    def __init__(self, model_path,
                 class_names,
                 keep_names,
                 nms_thr=0.45,
                 score_thr=0.3,
                 imgsz=(640, 640),
                 fp16=False
                 ):
        """
        :param model_path:  模型名称
        :param class_names: 模型标签，例如 ["person", "head"]
        :param keep_names:  需保留的标签 ["person"]
        :param nms_thr: nms阀值
        :param score_thr: box框的置新度
        :param imgsz:模型推理输入大小(640, 640)
        :param fp16:是否是fp16模型
        """

        self.nms_thr = nms_thr
        self.score_thr = score_thr

        # self.yolo_model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        self.yolo_model = onnxruntime.InferenceSession(model_path,
                                                       providers=['CUDAExecutionProvider', "CPUExecutionProvider"])
        self.input_name = [i.name for i in self.yolo_model.get_inputs()][0]
        self.output_name = [i.name for i in self.yolo_model.get_outputs()]
        self.class_names = class_names  # ["head", "people"]
        self.keep_names = keep_names
        self.fp16 = fp16
        self.imgsz = imgsz

    def run(self, img, nms_thr=None, score_thr=None):
        origin_img = np.copy(img)

        nms_thr = self.nms_thr if nms_thr is None else nms_thr
        score_thr = self.score_thr if score_thr is None else score_thr
        if nms_thr < 0.1:
            nms_thr = self.nms_thr

        if score_thr < 0.1:
            score_thr = self.score_thr

        img, ratio = pre_yolo(img, self.imgsz, None, None)
        if self.fp16:
            img = np.ascontiguousarray(img, dtype=np.float16)
        else:
            img = np.ascontiguousarray(img, dtype=np.float32)

        if self.fp16:
            predictions = self.yolo_model.run(self.output_name, {self.input_name: np.array([img])})
        else:
            predictions = self.yolo_model.run(None, {self.input_name: np.array([img])})
        predictions = predictions[0][0]
        predictions = predictions.transpose()
        predictions = predictions.astype("float32")
        boxes = predictions[:, :4]
        # scores = predictions[:, 4:5] * predictions[:, 5:]
        scores = predictions[:, 4:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
        boxes = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                cls_id = min(1,int(final_cls_inds[i]))
                score = final_scores[i]
                x0 = max(0, int(box[0]))
                y0 = max(0, int(box[1]))
                x1 = int(box[2])
                y1 = int(box[3])
                if (x1 - x0) > 3 and (y1 - y0) > 3:
                    boxes.append(
                        {"type": self.class_names[cls_id],
                         "position": [x0, y0, x1, y1],
                         "confidence": round(score, 2)
                         })

        boxes = [box for box in boxes if box['type'] in self.keep_names]

        return boxes

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "Z:\DL_Code\短裤短袖检测\短裤短袖Code\yahei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return np.asarray(img)


def vis(image,boxes):
    image = image.copy()
    for box in boxes:
        boxes = box['position']
        classify = box['type']
        color = {'pine_good':(0,255,255),'pine_bad':(0,0,255)}
        cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), color[classify], 3)
        # cv2.putText(image, str(classify), (int(boxes[0]), int(boxes[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color[classify], 2)
        # image = cv2ImgAddText(image,result_name[classify],int(boxes[0]),int(boxes[1])-20,color[classify],15)
    return image



if __name__ == '__main__':
    model_path = "model_files/pine.onnx"
    model = onnxModel(model_path,
                      class_names=['pine_good','pine_bad'],
                      keep_names=['pine_good','pine_bad'],)

    '''test single image'''
    path = "data/images/2023-04-17-shumu0_1_11.jpg"
    img = cv2.imread(path)
    boxes = model.run(img, score_thr=0.5)  ##保留run方法，用于其他程序调用,他过滤逻辑可以重载run方法
    print(boxes)
    show_image = vis(img,boxes)
    cv2.imshow('show image',show_image)
    cv2.waitKey()


    # '''test video'''
    # video_dir = './test_data/video2.mp4'
    # save_dir = './test_data/video2_result.mp4'
    # cap = cv2.VideoCapture(video_dir)
    # frame_num = 0
    # '''save video'''
    # fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    # ## 保存视频的参数
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
    # width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    # video_save = cv2.VideoWriter(save_dir, fourcc, fps, (width, height))  # 写入视频
    # success = True
    # while success:
    #     success, img = cap.read()
    #     if frame_num % 4 == 0:
    #         boxes = model.run(img, score_thr=0.5)  ##保留run方法，用于其他程序调用,他过滤逻辑可以重载run方法
    #     print(boxes)
    #     show_image = vis(img,boxes)
    #     video_save.write(show_image)
    #     frame_num += 1
    #     # cv2.imshow('show image',show_image)
    #     # cv2.waitKey()
    # cap.release()







