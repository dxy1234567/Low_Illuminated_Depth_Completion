import matplotlib.pyplot as plt

# 提取的 train 和 val loss 值
train_losses = [
    110.68920202, 100.97770756, 92.31793990, 89.50053508, 89.80340209, 
    84.74938058, 84.31680535, 79.04433401, 78.55796015, 79.05989802, 75.16373302, 
    74.66190666, 74.42391957, 74.08361915, 74.49886732, 72.61003709, 72.28348302, 
    72.37634124, 71.67019622, 72.15525035, 71.71991360, 71.24868811, 70.49161242, 
    69.35865454, 69.91441938, 69.52180937
]

val_losses = [
    82.31975209, 79.47862361, 78.04019687, 76.50759534, 78.89058001, 
    81.67496587, 73.13315639, 70.97532256, 69.77932921, 72.53060937, 69.92281958, 
    68.71704070, 68.88210069, 73.04948960, 69.04100023, 68.23233144, 67.37592196, 
    67.71502552, 68.47317375, 67.29995233, 69.24927198, 66.82222909, 67.96888960, 
    66.58133272, 67.88380754, 71.59030254
]

N = len(train_losses)


# 绘制折线图
epochs = list(range(1, N + 1))

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
plt.savefig('loss_plot3.png')