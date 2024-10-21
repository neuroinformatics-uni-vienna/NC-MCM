import tensorflow as tf
import numpy as np


class ScaleInvariantMSE(tf.keras.losses.Loss):
    def __init__(self, name="scale_invariant_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Ensure numerical stability by adding a small epsilon before taking the log
        epsilon = tf.keras.backend.epsilon()

        # Compute the logarithms of the true and predicted values
        log_y_true = tf.math.log(tf.abs(y_true) + epsilon)
        log_y_pred = tf.math.log(tf.abs(y_pred) + epsilon)

        # Compute the first term: mean squared error in log space
        log_mse = tf.reduce_mean(tf.square(log_y_pred - log_y_true), axis=-1)

        # Compute the second term: squared mean of log differences
        log_diff_mean = tf.reduce_mean(log_y_pred - log_y_true, axis=-1)
        # log_diff_mean_sq = tf.square(log_diff_mean)

        # Compute the scale-invariant MSE
        scale_invariant_mse = log_mse #- log_diff_mean_sq

        return scale_invariant_mse





# Test 1:
y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
y_pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 2:
y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
y_pred = np.array([2.0, 4.0, 6.0], dtype=np.float32)
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 3:
y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
y_pred = np.array([1.1, 2.2, 3.3], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 4:
y_true = np.array([10., 20., 30.], dtype=np.float32)
y_pred = np.array([11., 22., 33.], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 5:
y_true = 5*np.array([10., 20., 30.], dtype=np.float32)
y_pred = 5*np.array([11., 22., 33.], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 6:
y_true = np.array([10., 20., 30.], dtype=np.float32)
y_pred = np.array([1., 4., 6.], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 7:
y_true = 5*np.array([10., 20., 30.], dtype=np.float32)
y_pred = 5*np.array([1., 4., 6.], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 8:
y_true = 10*np.random.random((100,3))
y_pred = 10*np.random.random((100,3)) + np.random.random((100,3))
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 9:
y_true = -y_true
y_pred = -y_pred
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 10:
y_true = np.array([10., 20., 30.], dtype=np.float32)
y_pred = np.array([11., 22., 33.], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")

# Test 11:
y_true = 100 + np.array([10., 20., 30.], dtype=np.float32)
y_pred = 100 + np.array([11., 22., 33.], dtype=np.float32)  # Small uniform error
scale_invariant_mse =  ScaleInvariantMSE()(y_true, y_pred)
mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
print(f"scale_inv_loss = {scale_invariant_mse}, mse = {mse}")