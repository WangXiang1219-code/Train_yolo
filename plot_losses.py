import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_losses():
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
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制训练损失
    plt.subplot(2, 2, 1)
    plt.plot(data['epoch'], data['train/box_loss'], label='Box Loss')
    plt.plot(data['epoch'], data['train/cls_loss'], label='Class Loss')
    plt.plot(data['epoch'], data['train/dfl_loss'], label='DFL Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制验证损失
    plt.subplot(2, 2, 2)
    plt.plot(data['epoch'], data['val/box_loss'], label='Box Loss')
    plt.plot(data['epoch'], data['val/cls_loss'], label='Class Loss')
    plt.plot(data['epoch'], data['val/dfl_loss'], label='DFL Loss')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制mAP指标
    plt.subplot(2, 2, 3)
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.title('mAP Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    # 绘制精确率和召回率
    plt.subplot(2, 2, 4)
    plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision')
    plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join("fish_models", latest_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"损失函数图像已保存到: {plot_path}")

if __name__ == "__main__":
    plot_losses()