import os
import cv2
import torch
from ultralytics import YOLO

# 鱼类名称映射字典
CLASS_MAPPING = {
    'Bangus': '鲮鱼',
    'Big Head Carp': '鳙鱼',
    'Black Spotted Barb': '黑斑鲃',
    'Catfish': '鲶鱼',
    'Climbing Perch': '攀鲈',
    'Fourfinger Threadfin': '四指马鲅',
    'Freshwater Eel': '淡水鳗',
    'Glass Perchlet': '玻璃鲈',
    'Goby': '虾虎鱼',
    'Gold Fish': '金鱼',
    'Gourami': '斗鱼',
    'Grass Carp': '草鱼',
    'Green Spotted Puffer': '绿斑河豚',
    'Indian Carp': '印度鲤',
    'Indo-Pacific Tarpon': '印太骨舌鱼',
    'Jaguar Gapote': '美洲虎鱼',
    'Janitor Fish': '清道夫鱼',
    'Knifefish': '刀鱼',
    'Long-Snouted Pipefish': '长吻 pipefish',
    'Mosquito Fish': '食蚊鱼',
    'Mudfish': '泥鱼',
    'Mullet': '鲻鱼',
    'Pangasius': '巴沙鱼',
    'Perch': '鲈鱼',
    'Scat Fish': '弹涂鱼',
    'Silver Barb': '银鲃'
}

def load_model(model_path=None):
    """
    加载训练好的模型
    如果没有指定模型路径，则使用fish_models目录下的最新训练模型
    """
    if model_path is None:
        # 查找fish_models目录下的最新训练模型
        model_dirs = [d for d in os.listdir("fish_models") if d.startswith("fish_detection")]
        if model_dirs:
            # 按照修改时间排序，获取最新的目录
            latest_dir = max(model_dirs, key=lambda d: os.path.getmtime(os.path.join("fish_models", d)))
            model_path = os.path.join("fish_models", latest_dir, "weights", "best.pt")
            print(f"使用最新训练模型: {model_path}")
        else:
            raise FileNotFoundError("未找到训练好的模型文件")
    
    model = YOLO(model_path)
    return model

def predict_fish(image_path, model):
    """
    对图片中的鱼进行识别
    """
    # 进行预测
    results = model(image_path)
    
    # 获取结果
    result = results[0]
    
    # 获取类别ID和置信度
    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
    confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
    
    # 获取原始类名
    names = result.names
    
    # 转换为中文名
    fish_names = []
    for class_id, confidence in zip(classes, confidences):
        english_name = names[int(class_id)]
        chinese_name = CLASS_MAPPING.get(english_name, english_name)  # 如果没有中文名则使用英文名
        fish_names.append((english_name, chinese_name, float(confidence)))
    
    return fish_names

def draw_predictions(image_path, model, output_path='detect_result/detect_result.jpg'):
    """
    在图片上绘制预测结果并保存
    """
    # 确保result目录存在
    result_dir = os.path.dirname(output_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 读取图片
    image = cv2.imread(image_path)
    
    # 进行预测
    results = model(image_path)
    result = results[0]
    
    # 绘制预测框
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names
        
        for box, class_id, confidence in zip(boxes, classes, confidences):
            # 获取英文名
            english_name = names[int(class_id)]
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签（显示英文名和置信度）
            label = f'{english_name}: {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 保存结果图片
    cv2.imwrite(output_path, image)
    return output_path

def main():
    print("鱼类识别Demo")
    print("=" * 30)
    
    # 加载模型
    print("正在加载模型...")
    model = load_model()
    print("模型加载完成!")
    
    # 示例图片路径
    image_path = input("请输入鱼的图片路径 (或按回车使用默认图片): ").strip()
    
    if not image_path:
        # 如果没有提供图片路径，检查是否有默认图片
        test_images = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_images:
            image_path = test_images[0]
            print(f"使用默认图片: {image_path}")
        else:
            print("未找到图片文件，请提供有效的图片路径")
            return
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件未找到: {image_path}")
    
    # 进行预测
    print("正在进行鱼类识别...")
    fish_names = predict_fish(image_path, model)
    
    # 显示结果
    print("\n识别结果:")
    print("-" * 30)
    if fish_names:
        for english_name, chinese_name, confidence in fish_names:
            print(f"鱼类名称: {english_name} ({chinese_name}), 置信度: {confidence:.2f}")
        
        # 绘制结果，保存到result文件夹
        output_path = draw_predictions(image_path, model, 'detect_result/detect_result.jpg')
        print(f"\n结果图片已保存为: {output_path}")
    else:
        print("未检测到鱼类")

# 为了解决Windows控制台编码问题，我们直接运行main函数
if __name__ == "__main__":
    # 设置环境变量以支持UTF-8编码
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    main()