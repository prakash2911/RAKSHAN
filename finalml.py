
# Load saved Model
from keras.models import load_model

model = load_model('C:/Users/HP/Desktop/CNN_Cat_Dog_Model.h5')


# Part 3 - Making new predictions

# Place a new picture of a cat or dog in 'single_prediction' folder and see if your model works
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/HP/Desktop/Accidents/fire/8b4c1d994e747fdf4ebe96590e5c5ac2--smoke-damage-fire-safety.jpg', target_size = (64, 64))
# Add a 3rd Color dimension to match Model expectation
#C:/Users/HP/Desktop/Accidents/fire/8b4c1d994e747fdf4ebe96590e5c5ac2--smoke-damage-fire-safety.jpg
#C:/Users/HP/Desktop/Accidents/car/_b366f75e-0174-11eb-b32f-32d5f7e2c720.jpg
test_image = image.img_to_array(test_image)
# Add one more dimension to beginning of image array so 'Predict' function can receive it (corresponds to Batch, even if only one batch)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)
# We now need to pull up the mapping between 0/1 and cat/dog
#training_set.class_indices
# Map is 2D so check the first row, first column value
if result[0][0] == 0:
    prediction = 'car accident'
else:
    prediction = 'fire accident'
# Print result

print("\nPrediction: " + prediction)
