import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image, ImageFont, ImageDraw

# 鱼类英文名到中文名的映射
CLASS_MAPPING = {
    'AngelFish': '天使鱼',
    'BlueTang': '蓝吊鱼',
    'ButterflyFish': '蝴蝶鱼',
    'ClownFish': '小丑鱼',
    'GoldFish': '金鱼',
    'Fighting Fish': '斗鱼',
    'Scalar': '神仙鱼',
    'Moonlight Fish': '月光鱼',
    'RibbonedSweetlips': '条纹甜唇鱼',
    'ThreeStripedDamselfish': '三带雀鲷',
    'YellowCichlid': '黄慈鲷',
    'YellowTang': '黄吊鱼',
    'ZebraFish': '斑马鱼'
}

def load_model():
    """
    加载训练好的模型
    """
    # 使用指定的模型路径
    model_path = "fish_models/fish_detection/weights/best.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件未找到: {model_path}")
    
    # 加载模型
    model = YOLO(model_path)
    print(f"模型加载成功: {model_path}")
    return model, model_path

def predict_image(model, image_path, conf_threshold=0.5):
    """
    对单张图片进行预测
    """
    # 进行预测
    results = model(image_path, conf=conf_threshold)
    
    # 获取结果
    result = results[0]
    
    # 获取图像
    img = cv2.imread(image_path)
    
    # 绘制结果
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # 绘制每个检测框
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # 获取类别名
            class_name_en = result.names[class_id]
            class_name_cn = CLASS_MAPPING.get(class_name_en, class_name_en)
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签（只显示中文名和置信度）
            label_cn = f"{class_name_cn} {score:.2f}"
            
            # 计算标签尺寸
            (w_cn, h_cn), _ = cv2.getTextSize(label_cn, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 绘制标签背景
            cv2.rectangle(img, (x1, y1 - h_cn - 10), (x1 + w_cn, y1), (0, 255, 0), -1)
            
            # 绘制中英文标签
            # 使用支持中文的字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 如果需要中文支持，可以尝试以下方法
            # 使用OpenCV直接支持中文
            try:
                # 加载支持中文的字体文件
                font_path = "simsun.ttc"  # Windows系统中的宋体字体
                if not os.path.exists(font_path):
                    # 如果找不到simsun.ttc，尝试其他路径
                    font_path = "C:/Windows/Fonts/simsun.ttc"
                    if not os.path.exists(font_path):
                        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
                        if not os.path.exists(font_path):
                            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
                            if not os.path.exists(font_path):
                                font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
                                if not os.path.exists(font_path):
                                    font_path = None
                
                if font_path:
                    # 加载字体
                    font = ImageFont.truetype(font_path, 12)
                    # 创建PIL图像
                    pil_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    # 绘制中文文本
                    draw.text((x1, y1 - 5), label_cn, fill=(0, 0, 0), font=font)
                    # 转换回OpenCV格式
                    img = np.array(pil_img)
                else:
                    # 如果找不到字体，使用默认字体
                    cv2.putText(img, label_cn, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            except Exception as e:
                # 如果所有方法都失败，使用默认字体
                print(f"中文显示失败: {e}")
                cv2.putText(img, label_cn, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # # 绘制英文标签
            # if class_name_en:
            #     cv2.putText(img, class_name_en, (x1, y1 - 20),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img, result

def demo_single_image():
    """
    演示单张图片预测
    """
    # 加载模型
    model, model_path = load_model()
    
    # 创建测试图片目录
    test_images_dir = "test_images"
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
        print(f"请将要测试的图片放入 {test_images_dir} 目录中")
        return
    
    # 获取测试图片
    image_files = [f for f in os.listdir(test_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"在 {test_images_dir} 目录中未找到图片文件")
        print("请将要测试的图片放入该目录")
        return
    
    # 创建输出目录
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 对每张图片进行预测
    for image_file in image_files:
        print(f"处理图片: {image_file}")
        image_path = os.path.join(test_images_dir, image_file)
        
        # 预测
        result_img, result = predict_image(model, image_path)
        
        # 在控制台输出识别结果
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            print(f"图片 {image_file} 的识别结果:")
            for i, (score, class_id) in enumerate(zip(scores, class_ids)):
                class_name_en = result.names[class_id]
                class_name_cn = CLASS_MAPPING.get(class_name_en, class_name_en)
                print(f"  {i+1}. {class_name_cn} (置信度: {score:.2f})")
        
        # 保存结果
        output_path = os.path.join(output_dir, f"result_{image_file}")
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存到: {output_path}")

def main():
    """
    主函数
    """
    print("鱼类识别演示程序")
    print("开始处理 test_images 目录中的图片...")
    demo_single_image()

if __name__ == "__main__":
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()