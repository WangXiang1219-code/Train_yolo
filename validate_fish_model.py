import os
import torch
from ultralytics import YOLO

def validate_model():
    """
    验证训练好的模型
    """
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 查找最新的训练目录
    training_dirs = [d for d in os.listdir("fish_models") if d.startswith("fish_detection")]
    if training_dirs:
        # 按照修改时间排序，获取最新的目录
        latest_dir = max(training_dirs, key=lambda d: os.path.getmtime(os.path.join("fish_models", d)))
        model_path = os.path.join("fish_models", latest_dir, "weights", "best.pt")
        # model_path = "Model_fish_result/v8mfish1/fish_detection2/weights/best(v8mfishi1).pt"
        print(f"使用最新训练目录: {latest_dir}")
    else:
        raise FileNotFoundError("未找到任何训练目录")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件未找到: {model_path}")
    
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 定义数据集配置文件路径（使用与训练时相同的数据集）
    data_yaml_path = os.path.join(current_dir, "Data", "fish_datasets1", "data.yaml")

    
    # 验证模型
    print("开始验证模型...")
    # 降低批次大小以减少内存使用
    metrics = model.val(data=data_yaml_path, device=0, batch=8, imgsz=640)
    
    # 计算F1分数
    precision = metrics.box.mp
    recall = metrics.box.mr
    
    # 避免除零错误
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    # 打印关键指标
    print(f"验证完成！")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    
    return metrics

if __name__ == "__main__":
    # 设置环境变量以优化显存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    try:
        # 验证模型
        metrics = validate_model()
        print("模型验证结束")
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()