<div align="center">

# 컴퓨터 비전 5주차 실습
이번 실습은 Python과 OpenCV 라이브러리를 활용하여 <br>
MNIST 데이터셋을 이용한 다층 퍼셉트론(MLP) 기반 이미지 분류기 구현과, <br>
CIFAR-10 데이터셋을 활용한 CNN 모델 구축에 대한 실습입니다. <br>

</div>
<br><br>

# 1. 간단한 이미지 분류기 구현 (MNIST) (5.1.py)
[실습 목표]<br>
손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현합니다.
<br><br>
[과제 설명]<br>
tensorflow.keras.datasets에서 28x28 픽셀 크기의 흑백 이미지인 MNIST 데이터를 로드하고, <br> 
데이터를 훈련 세트와 테스트 세트로 분할합니다. <br>
Sequential 모델과 Dense 레이어를 결합하여 <br>
간단한 신경망 모델을 구축합니다. <br>
조기 종료(Early Stopping) 기법을 적용하여 <br>
모델을 훈련시키고 정확도를 평가합니다. <br>

<details>
<summary><b>전체 코드 및 주석 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
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
# [데이터 형태]
# x_train : [60000장]의 [28행 x 28열] 2차원 이미지 더미 (Shape: 60000, 28, 28)
# y_train : [60000개]의 정답 숫자 (예: 5, 0, 4, 1...) (Shape: 60000,)
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

# ==========================================
# 2. 데이터 전처리
# ==========================================
# [Flatten (평탄화) 및 정규화]
# 2D 이미지 (28x28)       =>  1D 배열 (784)            =>  정규화 (0.0 ~ 1.0)
# [  0,   0,   0]             [0, 0, 0, 255, 128...]   =>  [0.0, 0.0, 0.0, 1.0, 0.5...]
# [255, 128,   0]  ====>  
# [  0,   0,   0]  
x_train = x_train.reshape(60000, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(10000, 784).astype(np.float32) / 255.0

# [One-Hot Encoding]
# 정답 라벨이 숫자 '3'인 경우 => 인덱스 3만 1이고 나머지는 0인 배열로 변환
# 3  ====>  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (Shape: 60000, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ==========================================
# 3. 모델 구성
# ==========================================
# [신경망 데이터 흐름]
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
# [Early Stopping 작동]
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
plt.show()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1794" height="588" alt="image" src="https://github.com/user-attachments/assets/96182099-aac1-46ae-af80-ff0eaf42b37f" />
<img width="487" height="29" alt="image" src="https://github.com/user-attachments/assets/89241cfb-5646-474d-a685-ff6c8af16027" />
</div>
<br><br>

# 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축 (5.2.py)
[실습 목표] <br>
CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행합니다.
<br><br>
[과제 설명] <br>
tensorflow.keras.datasets에서 CIFAR-10 데이터셋을 로드한 뒤, 모델의 수렴이 빨라질 수 있도록 <br>
픽셀 값을 0~1 범위로 정규화하는 데이터 전처리를 수행합니다. <br>
2차원 공간 정보를 유지하기 위해 <br>
Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN 모델을 설계하고 훈련합니다. <br>
모델의 성능을 평가하고, 제공된 테스트 이미지(dog.jpg)에 대한 예측을 수행합니다. <br>
<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
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
    print(f"Error: '{img_path}' not found.")```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="528" height="236" alt="image" src="https://github.com/user-attachments/assets/6977602c-0df8-4a2e-a9e7-1f1da76497e3" />
</div>
<br><br>
