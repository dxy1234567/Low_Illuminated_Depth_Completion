from matplotlib import pyplot as plt

def plot_losses(losses, path_output):
    plt.plot(losses, label='Training Loss')

    # 设置标题和标签
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 保存图形到指定路径
    plt.savefig(path_output)