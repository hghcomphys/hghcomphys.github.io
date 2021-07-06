---
title: "Image Outlier Detection: Mahalanobis Distance"
categories:
  - Machine learning
tags:
  - Outlier detection
  - Deep learning
  - Image analysis
  - Keras
  - Python
header:
  image: &image "/assets/outlier-detection/images/outlier_mnist.png"
  caption: "MNIST Handwriting Digits & Outliers"
  teaser: *image
link: 
classes: wide
toc: true
toc_label: "Table of Contents"
# toc_icon: "cog"
author_profile: true
# layout: splash
---

<!-- I recently learned about outlier detection in context of medial image analysis.  -->
In this post, I will discuss how _Mahalanobis distance_ (MD) can be used to detect image outliers from [MNIST](http://yann.lecun.com/exdb/mnist/) handwriting digits dataset as an example.
Outliers are basically those sample images that don’t belong to a certain population in our training samples. 
Here we are particularly interested to identify those _known_ or _unknown_  populations that are out-of-distribution and are considered potentially as extreme cases for deployed machine learning models. 

Access to the entire script and more is available via this [jupyter notebook](https://github.com/hghcomphys/hghcomphys.github.io/blob/master/assets/outlier-detection/image_outlier_detection_mahalanobis_distance.ipynb).
{: .notice--info}

<!-- ## Background -->

### Importance of outlier detection
Deep neural networks (NNs) are powerful algorithms that have recently achieve impressive accuracies in various domains as wide as bioinformatics, disease diagnosis, waste management, and material novel material discovery. 
However, due to their intrinsic black box architecture, NNs exhibit an inclination to produce overconfident predictions and perform poorly on quantifying uncertainty. 
If an AI algorithm makes overconfident incorrect predictions then the consequences can be harmful or even 
catastrophic, for instance in the context of healthcare. 

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/outlier-detection/images/outlier_retina.png" alt="">
  <figcaption>An example outlier in retina images of diabetic retinopathy diagnosis which can cause an AI algorithm makes overconfident incorrect predictions.</figcaption>
</figure> 

This is exactly where outlier detection (OD) becomes crucial for practical applications by helping us to identify anomalies that are drawn far away from the distribution of the training samples and avoiding potentially wrong predictions.

**Note:** We are mainly interested in detecting out-of-distribution samples but those are not the only types of outliers. 
More about different types of outliers can be found [here](https://towardsdatascience.com/outliers-analysis-a-quick-guide-to-the-different-types-of-outliers-e41de37e6bf6).
{: .notice--info}

Outlier detection is in addition used to assess and improve the robustness of an AI model on out-of-distribution samples which results in increased generalization of developed AI models after deployment.  


### Mahalanobis distance
Assume having a $N$ dimensional multivariate space that $N$ basically indicates the number of variables. Each sample in this space is represented by a point and it is a vector $x=(x_1, x_2, ..., x_N)^T$.
Then the _Mahalanobis distance_ $D$ of any point $x$ is defined as the relative distance to the centroid of data distribution and it is given by 

$$
D(x) = \sqrt{ (x - \mu)^T V^{-1} (x - \mu) }, 
$$ 

where $\mu$ and $V$ are the `mean` and `covariance matrix` of the sample distribution, respectively.
The larger $D(x)$, the further away from the centroid the data point $x$ is.

A sample is identified as an outlier if its evaluated Mahalanobis distance, namely outlier score, is larger than a hard threshold obtained from the training inlier samples. 
The threshold, in this post, is defined as $\mu + 2 \sigma$ where $\sigma$ is the standard deviation.


### Feature space
<!-- Detecting outliers particularly in images is a difficult task. -->
Mahalanobis distance (MD) analysis can be directly applied to the image data.
It is however better to be applied on _feature space_ which is in fact an alternative representation of the original image data
in smaller dimensions encoded by a neural network. Any point in this space not only describing individual pixels but also intrinsically the appearance and relevant patterns for the labels present in the sample. 

The figure below shows a schematic view of a neural network which encodes an input images to a feature space. The feature extractor in fact is a representation of the original images but in a reduced dimension $x$.
<figure style="width: 500px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/outlier-detection/images/md_feature_model.png" alt="">
  <figcaption>A deep neural network as feature extractor maps an input image to a smaller feature space (x).</figcaption>
</figure> 

## Code description
In what follows, I will explain important steps on how to implement the MD outlier detection in python using `tensorflow.keras` which basically contains the data preparation, feature encoder, Mahalanobis outlier detector, and finally some results.


Access to the entire script and more is available via this [jupyter notebook](https://github.com/hghcomphys/hghcomphys.github.io/blob/master/assets/outlier-detection/image_outlier_detection_mahalanobis_distance.ipynb).
{: .notice--info}

### Data preparation
First of all, we rescale image data between `[-1.0, 1.0]` and intentionally using an [image data generator](https://keras.io/api/preprocessing/image/) from the input _numpy_ array.
The data generator is not necessary in this case as the size of the studied datasets is small (i.e. few megabytes). 
However, in the real-world problem, we often work with big data which the entire dataset can not be fitted to the memory at once. An image data generator is a great tool that helps us to deal with a vast number of images for different tasks such as preprocessing, augmentation, training, and validation.

#### Image data generator
The below script shows the definitions of image-related functions to create an `image data generator` from the input image (`X`) and label (`y`) data. 
<!--  -->
We use `image augmentations`, creating a modified versions of images by randomly applying a range of transformations, to reduce the chance of over-fitting while training the feature model.
_Keras_ `ImageDataGenerator` takes various image transformations (e.g. rightness, rotation, zoom) as input arguments. 
<!--  -->
The `one-hot` encoding is further used for the image labels.
The latter basically creates a binary column for each digit.  

```python
def scale(img):
    """
    Scale input image between [-1.0, 1.0].
    """
    return img.astype(np.float32)/127.5 - 1.0

def unscale(img):
    """
    Unscale input image between [0, 255].
    """
    return ((img+1.0)*127.5).astype(np.uint8)

def preprocess_image(img):
    """
    Preprocess function for image data generator.
    """
    img = scale(img)
    return img

def create_datagen(X, y, aug=False, onehot=False):
    """
    Create an image data generator from input images.
    """
    # Apply augmentation
    if aug:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=None,
            preprocessing_function=preprocess_image,
            shear_range=0.1,
            zoom_range=0.1,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rotation_range=10,
            brightness_range=[0.7, 1.3],
            fill_mode= "nearest",
        )   
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=None,
            preprocessing_function=preprocess_image,
        )
    
    # One-hot encodeing 
    labels = y if not onehot else tf.keras.utils.to_categorical(y)
    
    # Create data generator from numpy arrays (X, y).
    batches = datagen.flow(X, labels, batch_size=CFG['batch_size'], shuffle=True)
    batches.samples = X.shape[0]
        
    return batches
```

#### Datasets
The utility function `get_dataset()` helps us to easily load different datasets (i.e. `MNIST`, `fashion MNIST`, `CIFAR10`) for both feature model training and outlier detector validation.


```python
  def get_dataset(name="mnist"):
    """
    Return the entire dataset as a tuple (X, y).
    We need it to explicitly define the outlier population.
    """
    if name=="mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif name=="fashion_mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif name=="cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(f"Unknown dataset {name}")

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    # Add channel dimension for gray images
    if X.ndim < 4:
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
    
    return X, y
```

It might be asked why we need to concatenate the _training_ and _test_ datasets. 
The answer is that, only for testing purposes, we are interested in explicitly defined a _known_ outlier population (i.e. samples with label `9`, `8`). 
This can be done by separating them from the entire dataset and then splitting the rest into the training and test sets.
We further considering two `Fashion MNIST` and `CIFAR10` datasets as completely external set of _unknown_ outliers.
```python
X, y = get_dataset()

# Set some of categories as outliers and the rest for training classifier
outlier_indices = np.squeeze( (y==9) | (y==8) )
X_outliers, y_outliers = X[outlier_indices], y[outlier_indices]
X_inliers, y_inliers = X[~outlier_indices], y[~outlier_indices]

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X_inliers, y_inliers, test_size=0.20, random_state=2021)
num_labels = len(np.unique(y_train))
```
Bringing all together, we have eventually three sorts of data:
- training and test sets which contain image labels `0` to `7`
- _known_ outlier set which contains image labels `8` and `9`
- _unknown_ outliers belong to completely external datasets `Fashion MNIST` and `CIFAR10`

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/outlier-detection/images/datasets.png" alt="">
  <figcaption>Schematic representation of three different sorts of image data: training/test inliers (MNIST digits 0-7), known outliers (MNIST digits 8-9), and unknown outliers (Fashion MNIST and CIFAR10).</figcaption>
</figure> 

**Note:** The training and test sets have no samples from the outlier set. This means the later trained model doesn't know anything about outlier sets. This provides us a set of outliers to validate our outlier detector. 
{: .notice--info}


### Feature extractor model
For feature extractor model, we need to build a NN classifier model which contains two main parts:
- Feature extractor
- Top softmax layer

After training the classifier using provided image labels. we will use the feature extractor part which already contains the trained weights. This way, all the relevant features from training samples learned by the feature extractor and therefore is used for the outlier detection.


#### Building feature extractor
We build the feature extractor model by employing a deep neural network that contains a few sequences of `convolutional` layers and a `dense` layer on the top. 
The extended classifier model then is built using an additional dense layer with `softmax` activation functions. 

**INFO:** _Transfer learning_ is an alternative approach for developing a more advanced feature mapping instead of using the plain convolutional layers as discussing here.
{: .notice--info}

```python
def build_custom_model():
    """
    Build a convolutional classifier that is used as the image features extractor.
    """
    conv_regulizer = tf.keras.regularizers.l2(CFG['conv_regularization_factor'])
    dense_regulizer = tf.keras.regularizers.l2(CFG['dense_regularization_factor'])
    # Convolutional layers
    inp = layers.Input(shape=(CFG['image_size'], CFG['image_size'], CFG['channel_size']))
    x = inp
    for filters in CFG['layer_filters']:
        x = layers.Conv2D(filters, CFG['kernel_size'], padding='same', activation='relu', kernel_regularizer=conv_regulizer)(x)
        x = layers.AveragePooling2D((2, 2), padding='same')(x)
    # Top dense layers  
    x = layers.Flatten()(x)
    x= layers.Dropout(0.2)(x)
    features = layers.Dense(64, activation='sigmoid', kernel_regularizer=dense_regulizer)(x)   
    out = layers.Dense(num_labels, activation='softmax', kernel_regularizer=dense_regulizer)(features)
    # Return classifier, features extractor
    return Model(inp, out), Model(inp, features)
```

**Note:** Sigmoid activation function is used to map the output range of the feature extractor between [0, 1]. 
{: .notice--info}

Please note that the `build_custom_model()` function returns two models with the shared set of weights: 
1) classifier model 2) future extractor.


#### Fitting feature extractor
To find the right values of weights in the feature extractor model, the focus should be on the classifier model. We, therefore, compile the classifier using `adam` optimizer and `categorical cross-entropy loss function. Afterward, the training and validation batches are used to fit the classifier model.
```python
# Create NN which contains classifier and feature models
clf, fm = build_custom_model()

# Compile only the classifier, the one that has to be trained
clf.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Create training and validation image data generators
train_batches= create_datagen(X_train, y_train, aug=True, onehot=True)
val_batches = create_datagen(X_test, y_test, onehot=True)

# Fit classifier model
clf.fit(
    train_batches,
    steps_per_epoch=train_batches.samples//CFG['batch_size'],
    validation_data=val_batches,
    validation_steps=val_batches.samples//CFG['batch_size'],
    epochs=CFG['epochs'],
    callbacks=callbacks,
    workers=8,
    verbose=1,
 )
```

### Mahalanobis outlier detector
If everything goes well, we should eventually have a (well-)trained feature model which extracts important features from the input image and map them to the feature space. 
As the next step, we define a custom Mahalanobis outlier detector that accepts a pre-trained feature extractor model as an input, and having `fit` and `predict` methods. 
```python
class MahalanobisOutlierDetector:
    """
    An outlier detector which uses an input trained model as feature extractor and
    calculates the Mahalanobis distance as an outlier score.
    """
    def __init__(self, features_extractor: Model):
        self.features_extractor = features_extractor
        self.features = None
        self.features_mean = None
        self.features_covmat = None
        self.features_covmat_inv = None
        self.threshold = None
        
    def _extract_features(self, gen, steps, verbose) -> np.ndarray:
        """
        Extract features from the base model.
        """
        if steps is None:
            steps = gen.samples//CFG['batch_size']
        return self.features_extractor.predict(gen, steps=steps, workers=8, verbose=1)
        
    def _init_calculations(self):
        """
        Calculate the prerequired matrices for Mahalanobis distance calculation.
        """
        self.features_mean = np.mean(self.features, axis=0)
        self.features_covmat = np.cov(self.features, rowvar=False)
        self.features_covmat_inv = scipy.linalg.inv(self.features_covmat)
        
    def _calculate_distance(self, x) -> float:
        """
        Calculate Mahalanobis distance for an input instance.
        """
        return scipy.spatial.distance.mahalanobis(x, self.features_mean, self.features_covmat_inv)
    
    def _infer_threshold(self, verbose):
        """
        Infer threshold based on the extracted features from the training set.
        """
        scores = np.asarray([self._calculate_distance(feature) for feature in self.features])
        mean = np.mean(scores)
        std = np.std(scores)
        self.threshold = mean + 2 * std
        if verbose > 0:
            print("OD score mean:", mean)
            print("OD score std :", std)
            print("OD threshold :", self.threshold)  
            
    def fit(self, gen, steps=None, verbose=1):
        """
        Fit detector model.
        """
        self.features = self._extract_features(gen, steps, verbose)
        self._init_calculations()
        self._infer_threshold(verbose)
        
    def predict(self, gen, steps=None, verbose=1) -> np.ndarray:
        """
        Calculate outlier score (Mahalanobis distance).
        """
        features  =  self._extract_features(gen, steps, verbose)
        scores = np.asarray([self._calculate_distance(feature) for feature in features])
        if verbose > 0:
            print("OD score mean:", np.mean(scores))
            print("OD score std :", np.std(scores))
            print(f"Outliers     :{len(np.where(scores > self.threshold )[0])/len(scores): 1.2%}")
            
        if verbose > 1:
            plt.hist(scores, bins=100);
            plt.axvline(self.threshold, c='k', ls='--', label='threshold')
            plt.xlabel("Mahalanobis distance"); plt.ylabel("Distribution");
            plt.show()
            
        return scores
```

#### Fitting detector
Calling the `fit` method on training batches basically calculates the required `mean`, `covariance` matrix, and some statistics including `threshold` of Mahalanobis distance for inlier samples.
```python
# Create and fit Mahalanobis outlier detector
od = MahalanobisOutlierDetector(features_extractor=fm)
od.fit(train_batches, steps=None, verbose=1)
```

#### Detector predictions
The `predict` method calculates the Mahalanobis distances as outlier scores for any given input batches on sample level.
This allows us to calculate the outlier score for the training and validation sets using:
```python
od_scores =  od.predict(val_batches, verbose=2)
```

More interestingly, we can calculate the scores for the defined known outlier population.
```python
outlier_batches = create_datagen(X_outliers, y_outliers)
od_scores = od.predict(outlier_batches, verbose=2)
``` 

Samples from Fashion `Fashion MNIST` and `CIFAR10` datasets are considered as `unknown` outliers and using the `predict` method we can calculate their outlier scores. 


### Results and conclusions
We are eventually going to validate our fitted MD outlier detector on different populations of the MNIST dataset and two external datasets including Fashion MNIST and CIFAR10 to check how well it performs on the task aimed at detecting outliers. 
The obtained results are presented in the below table.

|       Subset       | Detected outlier | Mean score | Std score | Threshold | Samples |
|:------------------:|:----------------:|------------|-----------|:---------:|---------|
| MNIST (training)   |       2.52%      |    7.78    |    1.72   |   11.50   |  44960  |
| MNIST (validation) |       2.73%      |    7.51    |    1.71   |   11.50   |  11212  |
| MNIST (outlier)    |      23.26%      |    10.39   |    1.63   |   11.50   |  13760  |
| Fashion MNIST      |      59.62%      |    11.79   |    2.1    |   11.50   |  69968  |
| CIFAR10            |      24.74%      |    10.35   |    1.79   |   11.50   |  60000  |

Firstly, it is interesting to note that the Mahalanobis detector only detects `2.52%` of the population as outliers for the training MNIST samples. It expectedly gives a similar value of `2.73` for the validation samples.
However, it detects more outliers for the MNIST outlier, Fashion MNIST, and CIFAR10 datasets with corresponding values of `23.26%`, `59.62%`, and `24.74%`.
This tells us that the Mahalanobis OD enables us to detect outliers either they exist within the studied dataset or completely external datasets.
Secondly, the mean outlier score (i.e Mahalanobis distance) is found to be larger for outliers than those inliers of which used for training and test sets.
<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/outlier-detection/images/md_score_barplot.png" alt="">
  <figcaption>The mean outlier score for various populations of MNIST (training), MNIST(validation), MNIST (known outlier), Fashion MNIST (unknown outlier), and CIFAR10 (unknown outlier).</figcaption>
</figure> 


Further insights can be obtained by considering the distributions of the scores and focusing on the overlapping samples as shown in the next figure which is in fact the overlaid box and violin plots of the outlier score for different populations of MNIST (training), MNIST(validation), MNIST (known outlier), Fashion MNIST (unknown outlier), and CIFAR10 (unknown outlier). 
The figure reveals that there exist still some outliers that their score is small as compared with the inlier scores, Or samples within the training sets of which their scores is beyond the threshold.
This basically means that  Mahalanobis detector unable to identify all outliers but fraction of them and thus working to some extent.
<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/outlier-detection/images/md_score_boxplot.png" alt="">
  <figcaption>An overlaid box and violin plots of outlier scores for various populations of MNIST (training), MNIST(validation), MNIST (known outlier), Fashion MNIST (unknown outlier), and CIFAR10 (unknown outlier).</figcaption>
</figure>

The corresponding histogram plot of the outlier scores represents better the overlap regions.  
Nevertheless, detecting the shifts in data distribution due to outliers is pretty much obvious in the two previous  figures.
<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/outlier-detection/images/md_score_histplot.png" alt="">
  <figcaption>The corresponding histogram plot shows the shift in data distribution particularly for the Fashion MNIST population.</figcaption>
</figure> 
 

**INFO:** The jupyter notebook of the discussed script in this post can be found [here](https://github.com/hghcomphys/hghcomphys.github.io/blob/master/assets/outlier-detection/image_outlier_detection_mahalanobis_distance.ipynb).
{: .notice--info}


<!-- ### Conclusions
:) -->

### References
[https://arxiv.org/abs/1612.01474](https://arxiv.org/abs/1612.01474)