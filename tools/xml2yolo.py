import os
import xml.etree.ElementTree as ET
import glob

def convert_coordinates(size, box):
    """
    将XML中的边界框坐标转换为YOLO格式
    """
    dw = 1.0/size[0]
    dh = 1.0/size[1]

    # XML格式为 xmin, ymin, xmax, ymax
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    # 归一化
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)

def convert_xml_to_yolo(xml_path, class_mapping):
    """
    转换单个XML文件到YOLO格式
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 创建对应的txt文件路径
    txt_path = xml_path.replace('Annotations', 'Labels').replace('.xml', '.txt')

    # 确保Labels目录存在
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, 'w') as txt_file:
        # 处理每个目标
        for obj in root.iter('object'):
            # 获取类别名称
            class_name = obj.find('name').text

            # 获取类别ID
            if class_name not in class_mapping:
                continue
            class_id = class_mapping[class_name]

            # 获取边界框坐标
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)

            # 转换坐标
            bb = convert_coordinates((width,height), (xmin,ymin,xmax,ymax))

            # 写入txt文件
            txt_file.write(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

def main():
    # 定义类别映射
    class_mapping = {
        'missing_hole': 0,
        'mouse_bite': 1,
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5
    }

    # 获取所有XML文件
    xml_files = glob.glob('/home/pw/code/zsx/yolov11/ultralytics/dataset/PCB_DATASET/Annotations/*/*.xml')

    # 转换每个XML文件
    for xml_file in xml_files:
        try:
            convert_xml_to_yolo(xml_file, class_mapping)
            print(f"成功转换: {xml_file}")
        except Exception as e:
            print(f"转换失败 {xml_file}: {str(e)}")

if __name__ == "__main__":
    main()