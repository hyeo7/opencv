import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# [1] 데이터 로드 및 수치 정규화
# CIFAR-10 데이터를 불러오고 픽셀 값을 [0, 1] 범위로 스케일링하여 신경망의 연산 효율을 높임
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 레이블을 원-핫 인코딩(One-Hot Encoding)으로 변환하여 다중 클래스 분류 준비
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# [2] 데이터 증강 전략 수립
# 모델의 일반화 성능을 높이기 위해 이미지를 무작위로 변형하는 생성기 설정
datagen = ImageDataGenerator(
    rotation_range=15,       # 최대 15도 회전
    width_shift_range=0.1,   # 가로 10% 이동
    height_shift_range=0.1,  # 세로 10% 이동
    horizontal_flip=True,    # 좌우 반전 적용
)
datagen.fit(x_train)

# [3] 고성능 CNN 아키텍처 설계
model = Sequential()

# Block 1: 저수준 특징(선, 곡선) 추출 및 정규화
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', input_shape=(32,32,3)))
model.add(BatchNormalization()) # 배치 정규화로 학습 안정성 확보
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) # 과대적합 방지를 위한 20% 뉴런 비활성화

# Block 2: 중간 수준 특징(무늬, 질감) 추출
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# Block 3: 고수준 특징(형태) 추출
model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Classifier: 추출된 특징을 바탕으로 최종 분류
model.add(Flatten()) # 3차원 특징 맵을 1차원 벡터로 변환
model.add(Dense(128, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) # 파라미터가 많은 밀집층에 강력한 드롭아웃 적용
model.add(Dense(10, activation='softmax')) # 10개 클래스에 대한 확률 출력

# [4] 최적화 알고리즘 및 콜백 설정
opt = tf.keras.optimizers.Adam(learning_rate=0.0005) # 정교한 학습을 위해 학습률 조정
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 검증 손실이 개선되지 않으면 자동으로 중단하고 최적 가중치를 복구하는 콜백
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=12, 
    restore_best_weights=True 
)

# [5] 데이터 증강 흐름을 통한 학습 실행
print("\n--- Training Pipeline Started ---")
model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=100, 
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    verbose=2
)

# [6] 최종 일반화 성능 평가
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Final Test Accuracy: {accuracy * 100:.2f}%")

# [7] 외부 테스트 이미지(dog.jpg) 추론 및 검증
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img_path = 'dog.jpg'

if os.path.exists(img_path):
    # 이미지 로드 및 배열 변환
    img_raw = tf.keras.preprocessing.image.load_img(img_path)
    img_arr = tf.keras.preprocessing.image.img_to_array(img_raw)
    
    # Lanczos 알고리즘을 사용해 고품질 리사이즈 후 정규화
    img_input = tf.image.resize(img_arr, (32, 32), method='lanczos3')
    img_input = np.expand_dims(img_input.numpy(), axis=0) / 255.0
    
    # 모델 예측 수행
    pred = model.predict(img_input, verbose=0)
    conf = np.max(pred[0]) * 100
    p_class = class_names[np.argmax(pred[0])]
    
    print("\n" + "="*40)
    print(f"Prediction for '{img_path}': {p_class}")
    print(f"Confidence Score: {conf:.2f}%")
    print("="*40)
    
    # 목표 성능(80%) 달성 여부 확인
    if p_class == 'dog' and conf >= 80:
        print("Result: Objective Cleared (Over 80%)")
    else:
        print(f"Result: Misclassified as {p_class}")
else:
    print(f"Error: '{img_path}' not found.")
