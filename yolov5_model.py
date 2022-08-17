
import sys
sys.path.append("yolov5/")
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

import torch
import numpy as np

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Yolo5Detector():
    def __init__(self,weights,imgsz=(640, 640),device="cpu"):
        device = select_device("")
        self.model = DetectMultiBackend(weights, device=device, dnn=False,fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.img_size = imgsz[0]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000

    def detect_return_img(self,image):
        img0 = image
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        bs = 1  # batch_size

        im = torch.from_numpy(img).to(0)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        im0 = img0
        preds_json = []
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                s=""
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    preds_json.append(
                    {
                        "label":self.names[c],
                        "box":xyxy,
                        "score":conf
                    })
            
            im0 = annotator.result()
        return im0,preds_json

if __name__ == "__main__":
    model = Yolo5Detector("best.pt")
    img = cv2.imread("test/1.png")
    img,preds = model.detect_return_img(img)
    print(preds)
    cv2.imwrite("results/1.png",img)