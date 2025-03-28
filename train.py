import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('/home/pw/code/zsx/yolov11/ultralytics/improve/yolo11s-RFAConv.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data="/home/pw/code/zsx/yolov11/ultralytics/datasets/PKU-Market-PCB/data.yaml",
                # cache=True,
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=20,
                device='0',
                optimizer='Adam', # using SGD
                patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                workers=0,
                plots=True,
                amp=True,
                cos_lr=True,
                seed=143,
                project='runs/train-PKU-Market-PCB',
                name='yolov11s-RFAConv',
                )
