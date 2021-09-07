---
layout: single
title: "ensorFlow 2.x에서의 ImageDataGenerator로 데이터 증강 시 학습 안되는 현상 픽스"
---

# TensorFlow 2.x에서의 ImageDataGenerator

TensorFlow 1.x에서 ImageDataGenerator로 데이터 증강하여 학습 잘했었는데, 2.x에서 학습이 안되었다.<br>
이거 픽스한 기록.

<br>
Key Word : TensorFlow 2.x, Keras, ImageDataGenerator, image data augmentation, CIFAR10, image classification, CNN


# 문제 증상

다음은 기존 tensorflow 1.x에서 동작하던 코드 https://github.com/dhrim/mnd_advanced_2020/blob/master/material/deep_learning/cnn_cifar10.ipynb 에서 캡쳐 하였다.


```
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(raw_train_x, raw_train_y), (raw_test_x, raw_test_y) = tf.keras.datasets.cifar10.load_data()

print(raw_train_x.shape)
print(raw_train_y.shape)
print(raw_test_x.shape)
print(raw_test_y.shape)

train_x = raw_train_x/255
test_x = raw_test_x/255

train_y = raw_train_y
test_y = raw_test_y



datagen = ImageDataGenerator(
    rotation_range=10, # 0 ~ 180
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False 
)


model = keras.Sequential()
model.add(Input((32,32,3)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit_generator(datagen.flow(train_x, train_y, batch_size=128), epochs=5)


loss, acc = model.evaluate(test_x, test_y)
print("loss=",loss)
print("acc=",acc)

```

<br>

다음은 학습 시의 출력
```
Epoch 1/5
79/79 [==============================] - 16s 202ms/step - loss: 2.2660 - acc: 0.1197
Epoch 2/5
79/79 [==============================] - 15s 190ms/step - loss: 2.0844 - acc: 0.2221
Epoch 3/5
79/79 [==============================] - 15s 190ms/step - loss: 1.9601 - acc: 0.2758
Epoch 4/5
79/79 [==============================] - 15s 191ms/step - loss: 1.8455 - acc: 0.3251
Epoch 5/5
79/79 [==============================] - 15s 193ms/step - loss: 1.7525 - acc: 0.3521
10000/10000 [==============================] - 5s 460us/sample - loss: 1.6517 - acc: 0.3768
loss= 1.6517139757156372
acc= 0.3768
```


잘 동작한다.


이를 TensorFlow 2.x에서 돌렸더니, 학습이 영 안된다. loss는 떨어지는데 acc는 0.1에 머물러 있다. 

<br>

# 원인

TensorFlow 2.x로 업버전 하면서 내부 로직이 변경되었나 보다. 따로 문서로 명시된 것도 없고 이에 대한 문제 해결 방법도 못찾겠다.

하여간에 찾아낸 사실은 <br>

TensorFlow 2.x에서 ImageDataGenerator로 증강하려면
- Keras에서 제공하던 sparse 기능을 사용 못한다. one-hot encoding된 lable을 제공하여야 한다.
- Standardization 시켜야 학습이 진행된다. Normalization 말고 평균과 표준편차로 하는 Standardication.
<br>


# 수정한 사항

## category index가 아닌 one-hone 인코딩을 레이블링 값으로

레이블링 데이터의 값을 category index 정수값이였는데 이를 one-hoe 인코딩한 값으로 변경하였다.<br>
그리고 더불어 compile 시에 loss를 'sparse_categorical_crossentropy'가 아닌 'categorical_crossentropy'로 변경하였다.
```
train_y = to_categorical(train_y,10)
test_y = to_categorical(test_y,10)

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

## standardization

ImageDataGenerator 생성 시에 다음과 같이 standardization 옵션을 설정하였다.
```
datagen = ImageDataGenerator(
    ...
    samplewise_center=True,            # <----------------------------
    samplewise_std_normalization=True  # <----------------------------   
)
```
이 설정이 없으면 전혀 학습이 진행되지 않는다. loss는 떨어져도 acc가 0.1에 머문다. 



# 픽스된 코드

다음은 픽스된 코드 https://github.com/dhrim/cau_2021/blob/master/material/deep_learning/cnn_cifar10.ipynb 에서 캡쳐함.

```
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()


datagen = ImageDataGenerator(
    rotation_range=10, # 0 ~ 180
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False,
    samplewise_center=True,            # <----------------------------
    samplewise_std_normalization=True  # <----------------------------   
)



model = keras.Sequential()
model.add(Input((32,32,3)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))


train_y = to_categorical(train_y,10) # <----------------------------
test_y = to_categorical(test_y,10)   # <----------------------------

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   # <----------------------------
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])            # <----------------------------
model.summary()


model.fit(datagen.flow(train_x, train_y, batch_size=128), epochs=5)


loss, acc = model.evaluate(test_x, test_y)
print("loss=",loss)
print("acc=",acc)
```

다음은 위 코드의 학습 시의 출력.
```
Epoch 1/5
391/391 [==============================] - 39s 96ms/step - loss: 1.9809 - accuracy: 0.2638
Epoch 2/5
391/391 [==============================] - 37s 95ms/step - loss: 1.5669 - accuracy: 0.4348
Epoch 3/5
391/391 [==============================] - 37s 95ms/step - loss: 1.4029 - accuracy: 0.5001
Epoch 4/5
391/391 [==============================] - 37s 96ms/step - loss: 1.3328 - accuracy: 0.5258
Epoch 5/5
391/391 [==============================] - 37s 96ms/step - loss: 1.2932 - accuracy: 0.5442
313/313 [==============================] - 1s 4ms/step - loss: 53.2203 - accuracy: 0.3857
loss= 53.220333099365234
acc= 0.385699987411499
```









