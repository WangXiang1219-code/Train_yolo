import os
import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

def train_fish_model():
    """
    使用YOLOv11训练鱼类检测模型
    """
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 定义数据集配置文件路径
    data_yaml_path = os.path.join(current_dir, "Data", "fish_datasets1", "data.yaml")
    
    # 检查数据配置文件是否存在
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"数据配置文件未找到: {data_yaml_path}")
    
    # 加载预训练的YOLOv11模型
    # 可以选择不同大小的模型: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    model = YOLO("yolo11s.pt")  # 使用中等大小的模型平衡精度和性能
    
    # 设置训练设备
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个GPU")
        # 如果有多个GPU，可以选择只使用一个以避免共享内存问题
        if device_count > 1:
            device = 0  # 只使用第一个GPU
            print("检测到多个GPU，为避免共享内存问题，将只使用第一个GPU进行训练")
        else:
            device = 0  # 使用唯一的GPU
            print("使用GPU 0进行训练")
    else:
        device = None  # 使用CPU训练
        print("未检测到GPU，将使用CPU进行训练")
    
    # 开始训练模型
    results = model.train(
        data=data_yaml_path,      # 数据集配置文件路径
        epochs=100,               # 训练轮数
        imgsz=640,                # 图像尺寸
        batch=16,                 # 批次大小
        name="fish_detection",    # 实验名称
        project="fish_models",    # 项目目录
        device=device,            # 使用GPU设备进行训练
        cache=True,               # 开启图像缓存以加快训练速度
        workers=4,                # 数据加载线程数
        # amp=True,                 # 使用自动混合精度训练以加速训练
        # patience=30               # 提前停止的耐心轮数
    )
    
    print("训练完成！")
    
    # 绘制损失函数等图像
    plot_training_results()
    
    # 清理旧的训练目录，只保留最新的一个
    cleanup_old_training_dirs()
    
    return model, results

def plot_training_results():
    """
    绘制训练结果图表，包括损失函数、mAP等指标
    """
    # 查找最新的训练目录
    training_dirs = [d for d in os.listdir("fish_models") if d.startswith("fish_detection")]
    if not training_dirs:
        raise FileNotFoundError("未找到任何训练目录")
    
    # 按照修改时间排序，获取最新的目录
    latest_dir = max(training_dirs, key=lambda d: os.path.getmtime(os.path.join("fish_models", d)))
    results_path = os.path.join("fish_models", latest_dir, "results.csv")
    
    # 检查results.csv文件是否存在
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.csv文件未找到: {results_path}")
    
    # 读取训练结果数据
    data = pd.read_csv(results_path)

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 绘制训练损失
    plt.subplot(2, 3, 1)
    plt.plot(data['epoch'], data['train/box_loss'], label='Box Loss')
    plt.plot(data['epoch'], data['train/cls_loss'], label='Class Loss')
    plt.plot(data['epoch'], data['train/dfl_loss'], label='DFL Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制验证损失
    plt.subplot(2, 3, 2)
    plt.plot(data['epoch'], data['val/box_loss'], label='Box Loss')
    plt.plot(data['epoch'], data['val/cls_loss'], label='Class Loss')
    plt.plot(data['epoch'], data['val/dfl_loss'], label='DFL Loss')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制mAP指标
    plt.subplot(2, 3, 3)
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.title('mAP Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    # 绘制精确率和召回率
    plt.subplot(2, 3, 4)
    plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision')
    plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # 绘制学习率
    plt.subplot(2, 3, 5)
    plt.plot(data['epoch'], data['lr/pg0'], label='PG0')
    plt.plot(data['epoch'], data['lr/pg1'], label='PG1')
    plt.plot(data['epoch'], data['lr/pg2'], label='PG2')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    # 绘制总损失
    plt.subplot(2, 3, 6)
    train_total_loss = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
    val_total_loss = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
    plt.plot(data['epoch'], train_total_loss, label='Train Total Loss')
    plt.plot(data['epoch'], val_total_loss, label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join("fish_models", latest_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存
    
    print(f"训练结果图像已保存到: {plot_path}")

def cleanup_old_training_dirs():
    """
    删除 fish_models 目录下除最新训练目录外的所有旧目录
    """
    import shutil
    
    # 获取 fish_models 目录下的所有 fish_detectionX 子目录
    training_dirs = [d for d in os.listdir("fish_models") if d.startswith("fish_detection")]
    
    if len(training_dirs) <= 1:
        # 如果只有一个或没有目录，则无需清理
        return
    
    # 按照修改时间排序，获取最新的目录
    latest_dir = max(training_dirs, key=lambda d: os.path.getmtime(os.path.join("fish_models", d)))
    
    # 删除其他所有旧目录
    for dir_name in training_dirs:
        if dir_name != latest_dir:
            dir_path = os.path.join("fish_models", dir_name)
            shutil.rmtree(dir_path)
            print(f"已删除旧训练目录: {dir_path}")

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
    
    # 定义数据集配置文件路径
    data_yaml_path = os.path.join(current_dir, "Data", "fish_yolov11(1)", "data.yaml")
    
    # 验证模型
    metrics = model.val(data=data_yaml_path, device=0, batch=16, imgsz=640)
    
    # 打印关键指标
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return metrics

if __name__ == "__main__":
    # 设置环境变量以优化显存使用和避免共享内存问题
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print("开始训练鱼类检测模型...")

    # 训练模型
    trained_model, results = train_fish_model()

    # 验证模型
    # print("验证训练好的模型...")
    # validate_model()