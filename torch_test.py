import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    # 测试CUDA张量运算
    x = torch.rand(5, 3).cuda()
    print(f"CUDA张量测试: {x}")
else:
    print("CUDA不可用，仅使用CPU")