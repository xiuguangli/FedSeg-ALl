import torch
import numpy as np
import tensorflow as tf
from debug import BackCELoss_torch, BackCELoss_tf

class Args:
    num_classes = 3

def get_sample_data():
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)
    batch, h, w, num_classes = 2, 4, 4, 3
    logits_np = np.random.randn(batch, h, w, num_classes).astype(np.float32)
    labels_np = np.random.randint(0, num_classes, size=(batch, h, w)).astype(np.int64)
    # introduce some ignore labels
    labels_np[0, 0, 0] = 255
    return logits_np, labels_np

def test_loss():
    args = Args()
    logits_np, labels_np = get_sample_data()
    # torch version expects [N, C, H, W]
    logits_torch = torch.tensor(np.transpose(logits_np, (0, 3, 1, 2)), dtype=torch.float32)
    labels_torch = torch.tensor(labels_np, dtype=torch.int64)
    # tf version expects [N, H, W, C]
    logits_tf = tf.convert_to_tensor(logits_np, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels_np, dtype=tf.int64)

    torch_loss_fn = BackCELoss_torch(args)
    tf_loss_fn = BackCELoss_tf(args)

    torch_loss = torch_loss_fn(logits_torch, labels_torch).item()
    try:
        tf_loss = tf_loss_fn(logits_tf, labels_tf).numpy().item()
    except Exception as e:
        tf_loss = f"Error: {e}"

    print("Torch loss:", torch_loss)
    print("TF loss:   ", tf_loss)
    print("Diff:      ", "N/A" if isinstance(tf_loss, str) else abs(torch_loss - tf_loss))

    # Debug: check intermediate values
    print("\nDebug info:")
    print("labels_np unique:", np.unique(labels_np))
    print("logits_np shape:", logits_np.shape)

if __name__ == "__main__":
    test_loss()