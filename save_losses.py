import matplotlib.pyplot as plt

# 提取的 train 和 val loss 值
train_losses = [
    0.01204945, 0.00159072, 0.00152004, 0.00141552, 0.00153276,
    0.00148838, 0.00148854, 0.00146743, 0.00148889, 0.00149592,
    0.00142992, 0.00144495, 0.00143133, 0.00143031, 0.00142906,
    0.00141103, 0.00141377, 0.00140650, 0.00140927, 0.00141033
]

val_losses = [
    0.00131654, 0.00122353, 0.00117684, 0.00124701, 0.00124642,
    0.00127547, 0.00144639, 0.00123203, 0.00124027, 0.00126686,
    0.00127992, 0.00125950, 0.00122830, 0.00126870, 0.00122161,
    0.00125541, 0.00122973, 0.00122367, 0.00123715, 0.00130429
]

# 绘制折线图
epochs = list(range(1, 21))

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
plt.title('Train and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid()

# 保存图像
plt.savefig('loss_plot.png')
plt.show()