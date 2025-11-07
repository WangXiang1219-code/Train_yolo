from ultralytics import YOLO

# 加载预训练的YOLOv11模型
# 你可以根据需要选择不同大小的模型:
# - yolo11n.pt  # 最小、最快的模型
# - yolo11s.pt  # 小型模型
# - yolo11m.pt  # 中等大小模型
# - yolo11l.pt  # 大型模型
# - yolo11x.pt  # 最大、最精确的模型
model = YOLO("yolo11n.pt")

# 在你的鱼类数据集上训练模型
# 根据你的计算资源和时间要求，可以调整epochs参数
results = model.train(
    data="Data/fish_yolov11(1)/data.yaml",  # 数据集配置文件
    epochs=50,                              # 训练轮数 (可以根据需要调整)
    imgsz=640,                              # 图像大小
    batch=16,                               # 批次大小 (可以根据GPU内存调整)
    name="fish_detector"                    # 实验名称
)