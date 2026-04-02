import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# =============================================================================
# 1. CIFAR-10 데이터셋 로드
# 요구사항: CIFAR-10 데이터셋을 로드
# =============================================================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# =============================================================================
# 2. 데이터 전처리 (정규화)
# 요구사항: 데이터 전처리(정규화 등)를 수행
# =============================================================================
x_train, x_test = x_train / 255.0, x_test / 255.0

# =============================================================================
# 3. CNN 모델 설계 및 컴파일
# 요구사항: CNN 모델을 설계
# =============================================================================
model = models.Sequential([
    # 첫 번째 컨볼루션 블록 (특징 추출 및 크기 축소)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # 두 번째 컨볼루션 블록
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 세 번째 컨볼루션 블록
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 완전 연결층 (분류기)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10개 클래스 출력
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# =============================================================================
# 4. 모델 훈련
# 요구사항: 모델을 훈련
# =============================================================================
print("\n--- CNN 모델 학습 시작 ---")
# 훈련 속도를 고려하여 에포크를 10으로 설정 (필요시 조정)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# =============================================================================
# 5. 모델 성능 평가
# 요구사항: 모델의 성능을 평가
# =============================================================================
print("\n--- 모델 최종 성능 평가 ---")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"✅ 테스트 세트 정확도: {test_acc * 100:.2f}%")

# =============================================================================
# 6. 외부 테스트 이미지 (dog.jpg) 예측
# 요구사항: 테스트 이미지(dog.jpg)에 대한 예측을 수행
# =============================================================================
DOG_IMG_PATH = 'dog.jpg'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("\n--- 외부 이미지(dog.jpg) 추론 테스트 ---")
if os.path.exists(DOG_IMG_PATH):
    # 이미지 불러오기 및 (32, 32) 리사이즈
    img_raw = tf.keras.preprocessing.image.load_img(DOG_IMG_PATH, target_size=(32, 32))
    img_arr = tf.keras.preprocessing.image.img_to_array(img_raw)
    
    # 모델 입력 형태 (1, 32, 32, 3)에 맞게 차원 추가 및 정규화
    img_input = np.expand_dims(img_arr, axis=0) / 255.0
    
    # 예측 수행
    predictions = model.predict(img_input)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    print(f"👉 예측 결과: {predicted_class} (신뢰도: {confidence:.2f}%)")
else:
    print(f"⚠️ 에러: '{DOG_IMG_PATH}' 파일이 동일한 폴더에 존재하지 않습니다.")