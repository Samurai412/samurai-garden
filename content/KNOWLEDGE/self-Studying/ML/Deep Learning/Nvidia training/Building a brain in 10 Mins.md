Link:https://colab.research.google.com/github/NVDLI/notebooks/blob/master/building-a-brain/BuildingABrian.ipynb#scrollTo=4jUb-j_RU3cE

> [!faq] Fun FACT
> Many decades ago, artificial neural networks were developed to mimic the learning capabilities of humans and animals. Below is an excerpt from [The Machine that Changed the World](https://www.youtube.com/watch?v=enWWlx7-t0k&t=166s), a 1992 documentary about Artificial Intelligence.

https://youtu.be/cNxadbrN_aI


- Since then, computers and machine learning libraries have evolved to where we can replicate many days of experimentation in just a few minutes. In this notebook, we will step through how artificial neural networks have improved over the years and the biological inspiration behind it.

- To demonstrate, we will be using [TensorFlow](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.tensorflow.org%2F), an open-source machine learning library popular in industry. Recent versions of TensorFlow automatically detect if there is a GPU available for computation.


```python
import tensorflow as tf

  

tf.config.list_physical_devices('GPU')`
```
Output on Colab: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

- GPUs were originally designed for the significant amount of matrix mathematics used when rendering computer graphics. Neural networks also require a significant amount of matrix multiplication, making GPUs a good fit when building them.

- Speaking of graphics, we're going to tackle a challenge that seemed almost impossible decades ago: image classification with computer vision. Specifically, we will try to classify articles of clothing from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. A few samples are shown below:
![[Pasted image 20240516133633.png]]

- Neural networks attempt to copy the human learning technique, Trial and Error. To do this, we will create something like a set of digital flashcards. Our artificial brains will attempt to guess what kind of clothing we are showing it with a flashcard, then we will give it the answer, helping the computer learn from its successes and mistakes.

- Just like how students are quizzed to test their understanding, we will set aside a portion of our data to quiz our neural networks to make sure they understand the concepts we're trying to teach them, as opposed to them memorizing the answers to their study questions. For trivia, memorization might be an acceptable strategy, but for skills, like adding two numbers, memorization won't get our models very far.

 - The study data is often called the `training dataset` and the quiz data is often called the `validation dataset`. As Fashion MNIST is a popular dataset, it is already included with the TensorFlow library. Let's load it into our coding environment and take a look at it.

```python
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
```

> [!success] Explanation
> **`fashion_mnist = tf.keras.datasets.fashion_mnist`**
> 
>   
> * `tf.keras` refers to TensorFlow's Keras API, which provides an interface for building neural networks.
> * `datasets` is a module within Keras that contains pre-built datasets and utilities for working with them.
> * `fashion_mnist` is the specific dataset being loaded. It's a subset of the MNIST dataset, containing 70,000 images (55,000 training, 15,000 testing) of various fashion items (e.g., dresses, tops, pants).
> **`(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()`**
> 
> * `load_data()` is a method within the `fashion_mnist` dataset that loads and returns the training and validation data.
> * The assignment `(train_images, train_labels), (valid_images, valid_labels)` unpacks the returned values into four separate variables:
> + `train_images`: A 2D NumPy array containing the images for the training set. Each image is a grayscale pixel matrix with shape `(28, 28)`.
> + `train_labels`: A 1D NumPy array containing the corresponding labels (classifications) for each image in the training set.
> + `valid_images`: Similar to `train_images`, but contains the images for the validation set.
> + `valid_labels`: The corresponding labels for each image in the validation set.
>   
> In summary, this code loads a fashion-focused subset of the MNIST dataset and separates it into training and validation sets. This is often done when building machine learning models that require separate data for training and evaluating their performance.

- Let's start with our `train_images` and `train_labels`. `train_images` are like the question on our flashcards and `train_labels` are like the answer. In general, data scientists often refer to this answer as the `label`.

 - We can plot one of these images to see what it looks like. To do so, we will use [Matplotlib](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fmatplotlib.org%2F).

```python
import matplotlib.pyplot as plt
# The question number to study with. Feel free to change up to 59999.
data_idx = 42

plt.figure()

plt.imshow(train_images[data_idx], cmap='gray')

plt.colorbar()

plt.show()
```

> [!question] Explained
>  It creates a new figure using Matplotlib's `figure()` function. The default size is usually suitable for most plots.
> 
> After that, it displays an image from the dataset at the specified index (`data_idx`) and sets the color map (or colormap) used for rendering grayscale images to 'gray'.
>  
> Next, it adds a color bar to our plot, which shows how different pixel values correspond to specific colors. This is useful when working with grayscale or multi-channel images.
> 
> Finally, it displays the resulting image in a new window using Matplotlib's `show()` function.


# Building a Neuron


Neurons are the fundamental building blocks to a neural network. Just like how biological neurons send an electrical impulse under specific stimuli, artificial neural networks similarly result in a numerical output with a given numerical input.

  

We can break down building a neuron into 3 steps:

- Defining the architecture
- Intiating training
- Evaluating the model
## Defining the architecture

<center>

<a title="BruceBlaus, CC BY 3.0 &lt;https://creativecommons.org/licenses/by/3.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Blausen_0657_MultipolarNeuron.png"><img width="512" alt="Blausen 0657 MultipolarNeuron" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Blausen_0657_MultipolarNeuron.png/512px-Blausen_0657_MultipolarNeuron.png"></a>

<p><small>

Image courtesy of <a href="https://commons.wikimedia.org/wiki/File:Blausen_0657_MultipolarNeuron.png">Wikimedia Commons</a>

</small></p>

</center>


Biological neurons transmit information with a mechanism similar to [Morse Code](https://news.weill.cornell.edu/news/2007/09/scientists-find-clues-to-crack-brains-neural-code). It receives electrical signals through the dendrites, and under the right conditions, sends an electrical impulse down the axon and out through the terminals.

  

It is theorized the sequence and timing of these impulses play a large part of how information travels through the brain. Most artificial neural networks have yet to capture this timing aspect of biological neurons, and instead emulate the phenomenon with simpler mathematical formulas.

  
  

### The Math

  

Computers are built with discrete 0s and 1s whereas humans and animals are built on more continuous building blocks. Because of this, some of the first neurons attempted to mimic biological neurons with a linear regression function: `y = mx + b`. The `x` is like information coming in through the dendrites and the `y` is like the output through the terminals. As the computer guesses more and more answers to the questions we present it, it will update its variables (`m` and `b`) to better fit the line to the data it has seen.

  

Neurons are often exposed to multivariate data. We're going to build a neuron that takes each pixel value (which is between `0` and `255`), and assign it a weight, which is equivalent to our `m`. Data scientists often express this weight as `w`. For example, the first pixel will have a weight of `w0`, the second will have a weight of `w1`, and so on. Our full equation becomes `y = w0x0 + w1x1 + w2x2 + ... + b`.

  

Each image is 28 pixels by 28 pixels, so we will have a total of 784 weights. A pixel value of `0` would be black and a pixel value of `255` would be white. Let's look at the raw pixel values of the previous image we plotted. Each number below will be assigned a weight.

A single image looks roughly like this: `array([[  0,   0,   0,   0,   0,   0,   0,   0,  26,  10,   5,   5,   5,
          3,   4,   4,   3,   6,  24,   0,   0,   2,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0, 196, 203, 201, 234, 237,
        233, 231, 229, 196, 190, 207,  73,   0,   3,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   5, 195, 199, 194, 189, 192,
        188, 186, 186, 189, 180, 194,  67,   0,   1,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  32, 202, 195, 193, 186, 193,
        187, 186, 183, 189, 175, 197, 105,   0,   2,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  53, 208, 191, 190, 183, 190,
        188, 186, 184, 186, 167, 195, 132,   0,   1,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  75, 210, 191, 190, 184, 191,
        189, 184, 184, 185, 169, 190, 157,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 105, 209, 190, 191, 179, 190,
        188, 182, 184, 182, 170, 186, 177,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 135, 207, 188, 195, 180, 191,
        187, 180, 185, 181, 171, 184, 196,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 163, 205, 188, 197, 180, 191,
        185, 178, 187, 181, 172, 182, 203,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 179, 203, 183, 194, 180, 189,
        186, 180, 185, 179, 174, 179, 213,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 197, 199, 187, 188, 181, 185,
        187, 182, 184, 179, 175, 178, 185,  20,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 219, 194, 192, 189, 186, 184,
        187, 183, 185, 180, 177, 177, 192,  48,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 232, 192, 191, 188, 190, 183,
        186, 183, 185, 181, 178, 177, 195,  64,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  15, 237, 194, 194, 188, 192, 186,
        185, 187, 184, 183, 181, 178, 200,  83,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  52, 245, 199, 201, 194, 194, 191,
        203, 203, 191, 194, 195, 186, 206, 113,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  90, 239, 197, 201, 188, 193, 196,
        219, 218, 191, 188, 191, 183, 203, 146,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0, 101, 241, 195, 195, 183, 192, 187,
        209, 194, 186, 189, 185, 175, 197, 168,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0, 132, 234, 185, 195, 183, 195, 181,
        189, 193, 184, 193, 186, 181, 198, 192,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0, 176, 210, 195, 198, 191, 196, 187,
        188, 197, 187, 192, 188, 181, 196, 215,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0, 199, 208, 194, 195, 195, 195, 189,
        190, 195, 185, 194, 195, 183, 191, 221,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0, 191, 206, 206, 198, 196, 193, 191,
        194, 195, 189, 193, 197, 197, 193, 224,   0,   0,   0,   0,   0,
          0,   0],`


One more thing to think about: the output of `y = mx + b` is a number, but here, we're trying to classify different articles of clothing. How might we convert numbers into categories?

Here is a simple approach: we can make ten neurons, one for each article of clothing. If the neuron assigned to "Trousers" (label #1), has the highest output compared to the other neurons, the model will guess "Trousers" for the given input image.

[Keras](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkeras.io%2F), a deep learning framework that has been integrated into TensorFlow, makes such a model easy to build. We will use the [Sequential API](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkeras.io%2Fguides%2Fsequential_model%2F), which allows us to stack [layers](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkeras.io%2Fapi%2Flayers%2F), the list of operations we will be applying to our data as it is fed through the network.

In the below model, we have two layers:

- [Flatten](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Fkeras%2Flayers%2FFlatten) - Converts multidimensional data into 1 dimensional data (ex: a list of lists into a single list).
- [Dense](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Fkeras%2Flayers%2FDense) - A "row" of neurons. Each neuron has a weight (`w`) for each input. In the example below, we use the number `10` to place ten neurons.
![[Pasted image 20240516145705.png]] Dense illustration

> [!NOTE] How the above array looks when flattened
> [0 0 0 0 0 0 0 0 26 10 5 5 5 3 4 4 3 6 24 0 0 2 0 0 0 0 0 0 196 203 201 234 237 233 231 229 196 190 207 73 0 3 0 0 0 0 0 5 195 199 194 189 192 188 186 186 189 180 194 67 0 1 0 0 0 0]

We will also define an `input_shape` which is the dimensions of our data. In this case, our `28x28` pixels for each image.

```python
number_of_classes = train_labels.max() + 1

number_of_classes
```

```python
model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(number_of_classes)

])
```

> [!tip]
> **`tf.keras.Sequential`:**
> 
> * `Sequential` is a type of neural network architecture in Keras.
> 
> * It's called "sequential" because the layers are added one after another, forming a linear sequence.  
> 
> Think of it like building with blocks: each block (layer) is stacked on top of the previous one to form a tower. In this case, we're stacking two types of layers:
> 
> 
> 1. `Flatten`
> 
> 2. `Dense`
> 
> 
> **`tf.keras.layers.Flatten(input_shape=(28, 28))`:**
> 
> * The first layer in our sequence is a `Flatten` layer.
> * This layer takes the output from the previous convolutional and pooling layers (not shown here) and flattens it into a 1-dimensional array.
> * The `(28, 28)` input shape specifies that this layer expects an input with dimensions 28x28. In our case, these are the pixel values of an image.
>   
> Imagine taking a puzzle piece and stretching it out flat: that's what `Flatten` does to the output from earlier layers!
> 
> **`tf.keras.layers.Dense(number_of_classes)`:**
> 
> * The second layer in our sequence is a `Dense` (also known as fully connected or feedforward) layer.
> * This layer takes the flattened input and applies a linear transformation followed by an activation function.
> * The `number_of_classes` parameter specifies how many output neurons this layer should have. In our case, it's set to the number of classes we're trying to predict (e.g., 10 for digit recognition).
>   
> Think of it like building a bridge: each neuron in this layer is connected to all previous layers and applies an affine transformation followed by an activation function.


### Verifying the model

To make sure our model has the structure we expect, we can call the [summary](https://www.tensorflow.org/js/guide/models_and_layers#model_summary) method.
![[Pasted image 20240516150857.png]]
We can see that our total parameter count is `7850`. Let's see if this makes sense. For each pixel, there should be a weight for each of our ten classes.

So our weights make up `7,840` parameters. Where do the other ten come from? It's each of the `10` neurons biases, the `b` in `y = mx + b`.

  

There are a few other ways to verify our model. We can also [plot](https://keras.io/api/utils/model_plotting_utils/) it:

![[Pasted image 20240516151337.png]]

## Initiate Training

We have a model setup, but how does it learn? Just like how students are scored when they take a test, we need to give the model a function to grade its performance. Such a function is called the `loss` function.

In this case, we're going to use a type of function specific to classification called [SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy):
* **Sparse** - for this function, it refers to how our label is an integer index for our categories
* **Categorical** - this function was made for classification
* **Cross-entropy** - the more confident our model is when it makes an incorrect guess, the worse its score will be. If a model is 100% confident when it is wrong, it will have a score of negative infinity!
* `from_logits` - the linear output will be transformed into a probability which can be interpreted as the model's confidence that a particular category is the correct one for the given input.

This type of loss function works well for our case because it grades each of the neurons simultaneously. If all of our neurons give a strong signal that they're the correct label, we need a way to tell them that they can't all be right.

For us humans, we can add additional `metrics` to monitor how well our model is learning. For instance, maybe the loss is low, but what if the `accuracy` is not high?

```python
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
```

> [!important]
> **`optimizer='adam'`:**
> 
>   
> 
> * The `optimizer` parameter specifies how to update the model's weights during training.
> * In this case, we're using the Adam optimizer, which is a popular and effective algorithm for stochastic gradient descent (SGD).
> * Adam adapts learning rates based on the magnitude of gradients, making it suitable for deep neural networks.
> 
> Think of it like adjusting the speed of your car: Adam helps you find the optimal pace to learn from your mistakes!
> **`loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`:`**
>  
> * The `loss` parameter specifies how to measure the difference between predicted and actual outputs during training.
> * In this case, we're using the Sparse Categorical Cross-Entropy (SCCE) loss function with `from_logits=True`.
> + SCCE is a common choice for multi-class classification problems like image recognition.
> + The `from_logits=True` parameter means that our model will output raw logits (unnormalized scores) instead of probabilities. This allows us to use the softmax activation function later in the pipeline.
> 
> Think of it like calculating how far you are from your destination: SCCE measures the distance between predicted and actual class labels!
>  
> **`metrics=['accuracy']`:**
> 
> * The `metrics` parameter specifies what metrics to track during training.
> * In this case, we're tracking only one metric: accuracy (i.e., the proportion of correctly classified instances).
> + Accuracy is a common choice for evaluating classification models.
> 
> Think of it like checking your GPS navigation system: you want to know how accurate your predictions are!
> 
> By compiling our model with these settings, we set up the foundation for training and evaluating our neural network.


## Evaluating the model

Now the moment of truth! The below [fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) method will both help our model study and quiz it.

An `epoch` is one review of the training dataset. Just like how school students might need to review a flashcard multiple times before the concept "clicks", the same is true of our models.

After each `epoch`, the model is quizzed with the validation data. Let's watch it work hard and improve:


```python
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)
```

> [!NOTE]
> **`verbose=True`:**
> 
> * This controls whether the training process prints updates or not.
> 
> * In this case, we're setting `verbose` to True, which means our model will print progress messages during training.


`Epoch 1/5 1875/1875 [==============================] - 7s 3ms/step - loss: 16.9104 - accuracy: 0.7469 - val_loss: 11.2362 - val_accuracy: 0.7886 Epoch 2/5 1875/1875 [==============================] - 5s 2ms/step - loss: 12.1899 - accuracy: 0.7868 - val_loss: 18.4612 - val_accuracy: 0.6997 Epoch 3/5 1875/1875 [==============================] - 5s 3ms/step - loss: 11.4424 - accuracy: 0.7942 - val_loss: 10.8994 - val_accuracy: 0.8134 Epoch 4/5 1875/1875 [==============================] - 6s 3ms/step - loss: 10.4286 - accuracy: 0.8005 - val_loss: 16.7816 - val_accuracy: 0.7240 Epoch 5/5 1875/1875 [==============================] - 5s 3ms/step - loss: 10.7634 - accuracy: 0.8010 - val_loss: 10.8876 - val_accuracy: 0.7965`

How did the model do? B-? To give it credit, it only had `10` neurons to work with. Us humans have billions!

The accuracy should be around 80%, although there is some random variation based on how the flashcards are shuffled and the random value of the weights that were initiated.




### Prediction

Time to graduate our model and let it enter the real world. We can use the [predict](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict) method to see the output of our model on a set of images, regardless of if they were in the original datasets or not.

Please note, Keras expects a batch, or multiple datapoints, when making a prediction. To make a prediction on a single point of data, it should be converted to a batch of one datapoint.

Below are the predictions for the first ten items in our training dataset.