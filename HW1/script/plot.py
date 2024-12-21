# plot.py

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curves(output_dir):
    # 加载保存的数据
    train_losses = np.load(os.path.join(output_dir, 'train_losses.npy'))
    eval_exact_matches = np.load(os.path.join(output_dir, 'eval_exact_matches.npy'))

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Learning Curve of Loss Value')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # 绘制Exact Match曲线
    plt.figure(figsize=(10, 5))
    plt.plot(eval_exact_matches, label='Exact Match')
    plt.title('Learning Curve of Exact Match')
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'exact_match_curve.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot learning curves")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the .npy files and where to save the plots")
    args = parser.parse_args()
    plot_learning_curves(args.output_dir)