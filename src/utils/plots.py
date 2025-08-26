import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_training_curves(history, save_prefix):
    acc = history['train_acc']; val_acc = history['val_acc']
    loss = history['train_loss']; val_loss = history['val_loss']

    plt.figure()
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_acc.png', dpi=150)
    plt.close()

    plt.figure()
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_loss.png', dpi=150)
    plt.close()

def plot_confusion_matrix(cm, title='Confusion matrix', labels=None, save_path=None, normalize=True):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def plot_acc_vs_snr(snrs, acc_map, save_path):
    xs = list(snrs)
    ys = [acc_map[s] for s in xs]
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('CNN2 Classification Accuracy on RadioML 2016.10a')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
