from matplotlib import pyplot as plt

def plot_losses(train_losses, val_losses, path_output):
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
    plt.savefig(path_output)