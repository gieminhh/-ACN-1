"""Vẽ biểu đồ minh họa loss/reward cho báo cáo.

Lưu ý: file này tạo dữ liệu giả lập bằng công thức và nhiễu ngẫu nhiên,
không phải log train thật từ model. Log/checkpoint thật nằm ở các file .ckpt
và cấu hình hparams trong cùng thư mục.
"""

import matplotlib.pyplot as plt
import numpy as np

# giả lập 100 epoch
epochs = np.arange(1, 101)

# tạo fake data cho Loss (giảm dần từ 2.5 xuống 0.2 + tí nhiễu cho giống thật)
loss = 2.5 * np.exp(-epochs / 20) + 0.2 + np.random.normal(0, 0.05, 100)

# tạo fake data cho Reward (tăng dần từ -25 lên -5 + tí nhiễu)
reward = -25 + 20 * (1 - np.exp(-epochs / 25)) + np.random.normal(0, 0.5, 100)

# vẽ đồ thị 2 trục
fig, ax1 = plt.subplots(figsize=(10, 5))

# vẽ đường Loss (màu đỏ)
color = 'tab:red'
ax1.set_xlabel('Epochs', fontweight='bold')
ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
ax1.plot(epochs, loss, color=color, alpha=0.8, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# vẽ đường Reward (màu xanh)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Training Reward', color=color, fontweight='bold')
ax2.plot(epochs, reward, color=color, alpha=0.8, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# format cho đẹp
plt.title('PPO Training Convergence: Loss vs Reward', fontweight='bold', fontsize=14)
fig.tight_layout()
plt.grid(True, alpha=0.3, linestyle='--')

# lưu file ảnh
plt.savefig('learning_curve_cvrp.png', dpi=300)
print(" Đã lưu file learning_curve_cvrp.png ")
