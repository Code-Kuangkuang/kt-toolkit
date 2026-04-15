import torch
from mamba_ssm import Mamba

def test_mamba_operator():
    # 1. 检查 CUDA 是否可用（mamba_ssm 强依赖 CUDA）
    assert torch.cuda.is_available(), "错误：未检测到可用的 CUDA 设备！"
    device = "cuda"
    print(f"检测到显卡: {torch.cuda.get_device_name(0)}")

    # 2. 初始化一个极小的 Mamba 块
    batch_size, seq_len, dim = 2, 64, 16
    print("正在初始化 Mamba 模型...")
    model = Mamba(
        d_model=dim, # 模型维度
        d_state=16,  # 状态扩展因子
        d_conv=4,    # 局部卷积宽度
        expand=2,    # 块扩展因子
    ).to(device)

    # 3. 创建随机输入张量 (Batch, Length, Dimension)
    x = torch.randn(batch_size, seq_len, dim).to(device)
    print(f"输入张量形状: {x.shape}")

    # 4. 执行前向传播（这里会调用你刚才编译的 C++/CUDA 算子）
    print("正在执行前向传播...")
    y = model(x)

    # 5. 验证输出
    assert y.shape == x.shape, f"输出形状不匹配: 期望 {x.shape}, 实际 {y.shape}"
    print(f"输出张量形状: {y.shape}")
    print("🎉 测试完美通过！你的 mamba_ssm CUDA 算子运行正常！")

if __name__ == "__main__":
    test_mamba_operator()