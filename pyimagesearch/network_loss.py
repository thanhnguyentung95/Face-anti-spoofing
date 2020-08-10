import tensorflow as tf

def EDL(y_true, y_pred):
    tf.keras.backend.print_tensor(y_pred, 'y_pred')
    diff = tf.math.subtract(y_true, y_pred)
    squared_diff = tf.math.square(diff)
    squared_distance = tf.math.reduce_sum(squared_diff)
    distance = tf.math.sqrt(squared_distance)
    return distance
