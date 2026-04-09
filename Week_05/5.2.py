import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # [추가] 시각화를 위한 라이브러리 임포트
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# [요구사항 1] CIFAR-10 데이터셋 로드
# =============================================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# =============================================================================
# [요구사항 2] 데이터 전처리 (정규화 등) 수행
# =============================================================================
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 레이블 평탄화 (N, 1) -> (N,)
y_train = y_train.flatten()
y_test = y_test.flatten()

# =============================================================================
# [요구사항 3-1] CNN 모델 설계
# =============================================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # 완전 연결 층 파라미터 폭발 구간의 50%를 무작위 비활성화
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =============================================================================
# [요구사항 3-2] 모델 훈련 및 검증
# =============================================================================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, 
    restore_best_weights=True 
)

print("\n--- CNN 모델 학습 시작 ---")
# [수정] 학습 과정을 시각화하기 위해 반환값을 history 변수에 저장합니다.
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=50, 
    validation_split=0.1, 
    callbacks=[early_stopping],
    verbose=2
)

# =============================================================================
# [요구사항 4-1] 모델의 성능 평가
# =============================================================================
print("\n--- 테스트 세트 최종 평가 ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ 최종 테스트 정확도: {accuracy * 100:.2f}%")

# =============================================================================
# [부가 로직] 학습 곡선(Learning Curve) 시각화
# 과대적합 발생 시점 및 조기 종료 작동 여부를 시각적으로 검증합니다.
# =============================================================================
print("\n--- 학습 과정 시각화 그래프 출력 ---")
plt.figure(figsize=(12, 4))

# 1. 정확도(Accuracy) 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 2. 손실(Loss) 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss (Early Stopping Monitor)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# =============================================================================
# [요구사항 4-2] 테스트 이미지 (dog.jpg)에 대한 예측 수행
# =============================================================================
print("\n--- 외부 이미지 (dog.jpg) 예측 테스트 ---")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
img_path = 'dog.jpg'

if os.path.exists(img_path):
    img_raw = tf.keras.preprocessing.image.load_img(img_path)
    img_arr = tf.keras.preprocessing.image.img_to_array(img_raw)
    
    # 입력 비율을 유지하며 32x32로 맞추고 여백은 0으로 채움
    img_padded = tf.image.resize_with_pad(img_arr, 32, 32).numpy()
    img_input = np.expand_dims(img_padded, axis=0) / 255.0
    
    pred = model.predict(img_input, verbose=0)
    pred_index = np.argmax(pred[0])
    pred_class = class_names[pred_index]
    confidence = pred[0][pred_index] * 100
    
    print(f"👉 예측 결과: {pred_class} (신뢰도: {confidence:.2f}%)")
else:
    print(f"⚠️ 에러: '{img_path}' 파일이 동일한 폴더에 존재하지 않습니다.")