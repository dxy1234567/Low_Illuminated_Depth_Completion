import matplotlib.pyplot as plt

# 提取的 train 和 val loss 值
train_losses = [
    0.00288136, 0.00327892, 0.00203745, 0.00221372,
    0.00190913, 0.00177348, 0.00183075, 0.00187955, 0.00188626,
    0.00181815, 0.00180362, 0.00179548, 0.00171221, 0.00173819,
    0.00172128, 0.00166145, 0.00165876, 0.00163890, 0.00165656
]

val_losses = [
     0.00237730, 0.00197610, 0.00173435, 0.00177430,
    0.00162973, 0.00153087, 0.00152103, 0.00176938, 0.00164132,
    0.00161625, 0.00160920, 0.00150516, 0.00217914, 0.00190184,
    0.00147991, 0.00181515, 0.00150153, 0.00146034, 0.00143249
]


# 绘制折线图
epochs = list(range(1, 20))

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
plt.savefig('loss_plot1.png')
plt.show()