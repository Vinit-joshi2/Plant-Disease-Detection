# 🌿 Plant-Disease-Detection

A deep learning project for predicting plant diseases using a custom CNN built with TensorFlow/Keras.
The model classifies 38 types of plant diseases using the PlantVillage dataset.
Data preprocessing and augmentation techniques were applied to boost model performance.
Includes steps from loading and processing data to training and evaluating the model.

The PlantVillage dataset, originally published by Hughes & Salathé (2015), is a benchmark dataset for plant disease classification. It includes high-quality, expert-labeled images of both healthy and diseased plant leaves.

📊 Key Dataset Details
- Total Images: ~54,000 to 61,000 — depending on the version
  
- Classes: 38 distinct categories of plant species and disease combinations 

- Crop Types: Apple, Corn, Grape, Potato, Tomato, and more
  
📂 Dataset
- Source: <a href = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data"> Kaggle </a>

- Contains thousands of labeled images of healthy and diseased leaves from various crops.
  

This Porject demonstrates an end-to-end project for classifying plant diseases from leaf images.

- Objective: To build a deep learning model that can accurately identify 38 different types of plant diseases.

- Dataset: The public PlantVillage dataset.

- Approach: A custom Convolutional Neural Network (CNN) is built from scratch using TensorFlow/Keras. Data augmentation techniques are applied to improve model robustness.

- Workflow: The process includes data loading, preprocessing, model training, evaluation.

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/Images/img1.png">

<h1>
  Data Preprocessing
</h1>

<h4>
  
Let's see the leaf image with `256 x 256 x 3` size
</h4>

```
image_path = "/content/plantvillage-dataset/plantvillage dataset/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
img = plt.imread(image_path)
print(f" Image size - {img.shape}")
# Display the image
plt.imshow(img)
plt.show()

```
<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img3.png">

<h4>
  Sample of leaf image from Apple__Apple_scab folder
</h4>

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img4.png">

<h4>
  Now let's see Total Number of image from each folder
</h4>

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img5.png">

Distribution of our dataset

-  Orange__Haunglongbing category having 5507 leaf image
- Tomato__Tomato_yellow_leaf_curl_virus contain 5357 leaf image
- Potato_healthy contain lowest   leaf image which is around 152

<h1>
Data Augmentation
</h1>

<h4>
  
Why do we need data augmentation technique?
</h4>

In order to avoid overfiiting problem. we will `artifically` add data into our dataset through this we can make our existing data more larger.The idea is to alter the training data wth small transformation to reproduce the variation. Approaches that alter the training data in ways that change the array representation while keeping the label same , Some popular augmentations are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.,By applying data augmentation technique we can easily double the number of training data

<h4>
  
Common Data Augmentation Techniques
</h4>

- Rotation: Rotate the image by a certain angle (e.g., 90 degrees, 180 degrees).

- Translation: Shift the image horizontally or vertically by a certain distance.

- Scaling: Enlarge or shrink the image by a certain factor.

- Flipping: Flip the image horizontally or vertically.

- Shearing: Skew the image along the x or y-axis.

- Zooming: Zoom in or out of the image.

- Brightness Adjustment: Increase or decrease the brightness of the image.

- Contrast Adjustment: Increase or decrease the contrast of the image.

- Noise Addition: Add random noise to the image.

<h4>
  
For the data augmentation, i choosed to :
</h4>

- Randomly Zoom by 20% some training images

- Randomly flip images horizontally. Once our model is ready, we fit the training dataset.

- Vertical flip and Horizontal flip

```
datagen = ImageDataGenerator(
    rescale = 1./225 ,
    zoom_range = 0.2 ,
    shear_range = 0.2  ,
    vertical_flip = True ,
    horizontal_flip = True ,
    validation_split = 0.2
)

dir = "/content/plantvillage-dataset/plantvillage dataset/color"
# Train Generator
train_generator = datagen.flow_from_directory(
    dir ,
    target_size = (256 , 256) ,
    batch_size = 32 ,
    subset = "training",
    class_mode = "categorical"

)

# Vlidation Generator
validation_generator = datagen.flow_from_directory(
    dir ,
    target_size = (256 , 256) ,
    batch_size = 32 ,
    subset = "validation" ,
    class_mode = "categorical"
)

```

<h1>
Model
</h1>

<p>
Convolutional layers - 

- They are the fundamental building blocks of CNNs. These layers perform a critical mathematical operation known as convolution.

- This process entails the application of specialized filters known as kernels, that traverse through the input image to learn complex visual patterns.
  
</p>

<p>
Kernels -

- They are essentially small matrices of numbers. These filters move across the image performing element-wise multiplication with the part of the image they cover, extracting features such as edges, textures, and shapes.

- This is a basic example with a 2 × 2 kernel:
</p>

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img6.png">

<p>
We start in the left corner of the input:

- (0 × 0) + (1 × 1) + (3 × 2) + (4 × 3) = 19
  
Then we slice one pixel to the right and perform the same operation:

- (1 × 0) + (2 × 1) + (4 × 2) + (5 × 3 ) = 25
  
After we completed the first row we move one pixel down and start again from the left:

- (3 × 0) + (4 × 1) + (6 × 2) + (7 × 3) = 37
  
Finally, we again slice one pixel to the right:

- (4 × 0) + (5 × 1) + (7 × 2) + (8 × 3) = 43
  
The output matrix of this process is known as the Feature map.
</p>

<h4>
  Now let's see the basic architecture of cnn model
</h4>

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img7.png">

```
model = Sequential()
# Input size = 256 x 256
model.add(Conv2D(32 , (3,3) , activation = "relu" , input_shape = (256 , 256 , 3)))
# output size = 256 x 256
model.add(MaxPooling2D(2,2))
# output size = 128 x 128

# input size = 128 x 128
model.add(Conv2D(64 , (3,3) , activation = "relu"))
# output size = 128 x 128
model.add(MaxPooling2D(2,2))
# output size = 64 x 64



# The entire feature map and reorganizes it into a single, long vector.
model.add(Flatten())
model.add(Dense(38, activation="softmax"))


```

<p>
This is how our model look like -

- Taking the image as an input with 256 x 256 x 3 size
- Applying convolutional layer with 32 neurons
- Once image will pass through convolutional layer then applying maxpooling with 2 x2 (image size - 128 x 128)
- Again passing our image ro second convolutional layer with 64 neurons
- Again using maxpooling layer (image size - 64 x 64)
- Now entire feature are converting it into a single, long vector.
  
</p>

```
  # Model Evaluation
print("Evaluating model...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps= 120)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

```

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img8.png">

<p>
  Getting around 80% accuracy in test data , can't say truely good as well as truely bad , But still with 80% accuracy in test data our model quite working well. But still we can increse accuracy by using more epochs.Maybe around 50 epochs or more than that
</p>

<h4>
  Now let's see the graph of Train accuracy and Test accuracy , through this we will get to know how our model is working on train and test data
</h4>

```
# Plot training and validation accuracy value
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Train and Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train" , "Test"], loc = "upper left")
plt.show()

```

<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img9.png">

<p>
  With 10 epochs we are getting well accuracy on train and test accuracy , not facing overfiitig which is quiet good
</p>


<h4>
  Now let see loss graph of train ans test
</h4>

```
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and Test loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

```
<img src = "https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img10.png">

<p>
with epochs both train and test loss are decresing which is quiet good after 9 epochs our model is doing less error on train as well on test data.
But still we can try with more epochs to understand how our model is working like - [20 , 30 , 50 epochs ]
  
</p>

<h1>
Let Build a Predictive Model
</h1>

<p>
  <h4>How our model will work?</h4>
  
- Do some preprocessing and convert imnage into array
- Once we push any image as an input into our convolutional neurel network, our model try to  capture feature from image and give the prediction
- 
</p>

<p>Preprocessing Part</p>

```
# Function to load and Process the image
def load_and_preprocess_image(image_path , target_size = (256 , 256)):
  img = Image.open(image_path)
  # Resize the image
  img = img.resize(target_size)
  # cpnvert the image to a numpy  array
  img_array = np.array(img)
  # Add batch dimension
  img_array = np.expand_dims(img_array , axis = 0)
  # Scale the image value to [0,1]
  img_array = img_array.astype("float32") / 255,
  return img_array

```

<p>
  Prediction Part
</p>

```
# Function to predict the class of image
def predict_image_class(model , image_path , class_indeces):
  preprocessed_img = load_and_preprocess_image(image_path)
  predictions = model.predict(preprocessed_img)
  predicted_class_index = np.argmax(predictions , axis = 1)[0]
  predicted_class_name = class_indeces[predicted_class_index]

  return predicted_class_name


class_indices = {v :k for k , v in train_generator.class_indices.items()}
class_indices

```

<h4>Now let's test our model</h4>

```
image_path = "/content/plantvillage-dataset/plantvillage dataset/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
img = plt.imread(image_path)
plt.imshow(img)

predicted_class_name = predict_image_class(model , image_path , class_indices)
print("Predicted class Name :" , predicted_class_name)

```

<img src = 'https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img11.png'>

<p>AS you can see our model correctly predicting leaf disease</p>


```
image_path = "/content/plantvillage-dataset/plantvillage dataset/color/Blueberry___healthy/008c85d0-a954-4127-bd26-861dc8a1e6ff___RS_HL 2431.JPG"
img = plt.imread(image_path)
plt.imshow(img)

predicted_class_name = predict_image_class(model , image_path , class_indices)
print("Predicted class Name :" , predicted_class_name)


```

<img src = 'https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img12.png'>

```
image_path = "/content/plantvillage-dataset/plantvillage dataset/color/Peach___Bacterial_spot/002eddd0-b6b3-474c-be08-423e53e24f82___Rutg._Bact.S 1955.JPG"
img = plt.imread(image_path)
plt.imshow(img)
predicted_class_name = predict_image_class(model , image_path , class_indices)
print("Predicted class Name :" , predicted_class_name)

```

<img src = 'https://github.com/Vinit-joshi2/Plant-Disease-Detection/blob/main/img13.png'>
