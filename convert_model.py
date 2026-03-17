import tensorflow as tf

model = tf.keras.models.load_model("fall_detection_lstm.h5")
model.save("fall_detection_lstm_fixed.keras")

print("Model converted successfully")