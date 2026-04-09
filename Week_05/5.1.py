import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

# 2. 데이터 전처리
# 28x28 픽셀의 그레이스케일 이미지를 1차원 배열(784)로 평탄화 및 정규화(0~1)
x_train = x_train.reshape(60000, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(10000, 784).astype(np.float32) / 255.0

# 정답 레이블을 원-핫 인코딩(One-Hot Encoding)으로 변환
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. 모델 구성
dmlp = Sequential()
dmlp.add(Dense(units=128, activation='relu', input_shape=(784,)))
# 최소한의 일반화를 위해 가벼운 Dropout 추가
dmlp.add(Dropout(0.2)) 
dmlp.add(Dense(units=10, activation='softmax'))

dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 4. 지능적인 학습 제어 (Early Stopping)
# 검증 손실(val_loss)이 3번 연속 개선되지 않으면 학습 중단, 가장 좋았던 가중치로 복구
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 5. 모델 학습
# 최대 50번을 지시하지만, 조기 종료가 개입하여 과적합을 방지함
hist = dmlp.fit(
    x_train, y_train, 
    batch_size=128, 
    epochs=50, 
    validation_data=(x_test, y_test), 
    callbacks=[early_stopping],
    verbose=2
)

# 6. 모델 평가 및 저장
loss, accuracy = dmlp.evaluate(x_test, y_test, verbose=0)
print(f"\n최종 테스트 정확도 (Test Accuracy) = {accuracy * 100:.2f}%")

# 최신 Keras 권장 포맷으로 저장
dmlp.save('dmlp_optimized.keras')

# 7. 학습 과정 시각화
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