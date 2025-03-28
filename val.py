import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
from ultralytics.models.yolo.detect.val import DetectionValidator

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO


def val_end_callback(self: DetectionValidator):
    txt_path = os.path.join(self.save_dir, 'results.txt')
    p, r = self.metrics.mean_results()[:2]
    pf = "%11s" + "%18i" * 2 + "%18.8g" * len(self.metrics.keys) + '%18.8g\n' # print format
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(("%11s" + "%18s" *7 + '\n') % ("Class", "Images", "Instances", "Box(P", "Recall", "mAP50", "mAP50-95", "F1)"))
        f1 = 2 * (p * r ) / (p + r + 1e-16)
        f.write(pf % ("all", self.seen, self.nt_per_image.sum(), *self.metrics.mean_results(), f1))
        for i, c in enumerate(self.metrics.ap_class_index):
            p, r = self.metrics.class_result(i)[:2]
            f1 = 2 * (p * r) / (p + r + 1e-16)
            f.write(pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i), f1))


if __name__ == '__main__':
    model = YOLO('runs/train-1/WiderPerson-v8n-CAFM-CIoU/weights/best.pt')
    # model.info(detailed=True)
    model.add_callback('on_val_end', val_end_callback)
    model.val(
        data='/home/ubuntu/code/zsx/yolo-thesis/ultralytics-main/dataset/WiderPerson-1/data.yaml',
        # data='/home/ubuntu/code/zsx/yolo-thesis/ultralytics-main/dataset/zg-djdx/data.yaml',
        split='val',
        imgsz=640,
        batch=8,
        rect=False,
        seed=143,
        plots=True,
        half=True,
        # conf=0.001,
        save_json=True,
        iou=0.7,

        project='runs/val-1',
        name='WiderPerson-v8n-CAFM-CIoU',
        # project='runs/val-zg',
        # name='zg-djdx-v8n-smallhead-delete-large-GSConv-VoVGSCSP2',
        verbose=True,
        )