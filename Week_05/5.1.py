import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. 데이터 로드
# ==========================================
# [데이터 형태 시각화]
# x_train : [60000장]의 [28행 x 28열] 2차원 이미지 더미 (Shape: 60000, 28, 28)
# y_train : [60000개]의 정답 숫자 (예: 5, 0, 4, 1...) (Shape: 60000,)
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

# ==========================================
# 2. 데이터 전처리
# ==========================================
# [Flatten (평탄화) 및 정규화 시각화]
# 2D 이미지 (28x28)       =>  1D 배열 (784)            =>  정규화 (0.0 ~ 1.0)
# [  0,   0,   0]             [0, 0, 0, 255, 128...]   =>  [0.0, 0.0, 0.0, 1.0, 0.5...]
# [255, 128,   0]  ====>  
# [  0,   0,   0]  
x_train = x_train.reshape(60000, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(10000, 784).astype(np.float32) / 255.0

# [One-Hot Encoding 시각화]
# 정답 라벨이 숫자 '3'인 경우 => 인덱스 3만 1이고 나머지는 0인 배열로 변환
# 3  ====>  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (Shape: 60000, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ==========================================
# 3. 모델 구성
# ==========================================
# [신경망 데이터 흐름 시각화]
# Input Layer        Hidden Layer             Dropout               Output Layer
# (784개 픽셀)        (128개 노드, ReLU)      (20% 임의 차단)        (10개 클래스, Softmax)
# 
# [○] --가중치-->    [●]                      [●] --가중치-->      [◎] (0일 확률 2%)
# [○] --가중치-->    [●]  ====>               [X]  ====>           [◎] (1일 확률 5%)
# ...                ...                      ...                  ...
# [○] --가중치-->    [●]                      [●] --가중치-->      [◎] (9일 확률 90%) -> 9로 예측!
dmlp = Sequential()
dmlp.add(Dense(units=128, activation='relu', input_shape=(784,)))
dmlp.add(Dropout(0.2)) 
dmlp.add(Dense(units=10, activation='softmax'))

dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# ==========================================
# 4. 지능적인 학습 제어 (Early Stopping)
# ==========================================
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ==========================================
# 5. 모델 학습
# ==========================================
# [조기 종료 작동 시각화]
# Epoch 10: loss 0.05, val_loss 0.080 (최저점 갱신)
# Epoch 11: loss 0.04, val_loss 0.082 (악화 1/3)
# Epoch 12: loss 0.03, val_loss 0.085 (악화 2/3)
# Epoch 13: loss 0.02, val_loss 0.090 (악화 3/3) => 인내심(patience) 한계 도달! 학습 중단 및 Epoch 10 가중치로 롤백.
hist = dmlp.fit(
    x_train, y_train, 
    batch_size=128, 
    epochs=50, 
    validation_data=(x_test, y_test), 
    callbacks=[early_stopping],
    verbose=2
)

# ==========================================
# 6. 모델 평가 및 저장
# ==========================================
loss, accuracy = dmlp.evaluate(x_test, y_test, verbose=0)
print(f"\n최종 테스트 정확도 (Test Accuracy) = {accuracy * 100:.2f}%")

dmlp.save('dmlp_optimized.keras')

# ==========================================
# 7. 학습 과정 시각화
# ==========================================
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.grid(True)

plt.tight_layout()
plt.show()
