
from keras.models import load_model

model = load_model('C:/Users/HP/Desktop/CNN_Cat_Dog_Model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/HP/Desktop/Accidents/fire/8b4c1d994e747fdf4ebe96590e5c5ac2--smoke-damage-fire-safety.jpg', target_size = (64, 64))

#C:/Users/HP/Desktop/Accidents/fire/8b4c1d994e747fdf4ebe96590e5c5ac2--smoke-damage-fire-safety.jpg
#C:/Users/HP/Desktop/Accidents/car/_b366f75e-0174-11eb-b32f-32d5f7e2c720.jpg
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)

if result[0][0] == 0:
    prediction = 'car accident'
else:
    prediction = 'fire accident'

print("\nPrediction: " + prediction)
