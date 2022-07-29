# Artistic Style Transfer
This project is an extension of the [fast-neural-style](https://github.com/pytorch/examples/tree/main/fast_neural_style) example from pytorch. The main purpose is to test the pretrained mosaic model on a random image, create a new mosaic model using a new dataset and experiment with data augmentation and activation functions.

* [Setup](#Setup)
* [Tasks](#Tasks)
    * [Mosaic version of soccer-ball image](##Mosaic-version-of-soccer-ball-image)
    * [New Dataset](##New-Dataset)
        * [Train](#Train)
        * [Style](#Style)
    * [Data Augmentation](##Data-Augmentation)
        * [Train](#Train-1)
        * [Style](#Style-1)
    * [Activation function](##Activation-function)
        * [Train](#Train-2)
        * [Style](#Style-2)

 
# Setup 
MacOS Monterey was used to develop and run the models. As of the newest update, Pytorch uses Apple’s Metal Performance Shaders (MPS) as a backend, thus enabling high-performance training on GPU.

Therfore, a new argument is created in the **fast-neural-style** example **(*--mac-m1***) in order to enable training on GPU. The flag is either 1 or 0(default), indicating that we run on MacOS or not, respectively.


> **_NOTE:_**  The M1-GPU support feature is supported only in MacOS Monterey (12.3+).


# Tasks

In order to perform the tasks we first need to download the pretrained models. Inside the *fast_neural_style* directory run:
```
python download_saved_models.py
```
## Mosaic version of soccer-ball image

*The image can be found in fast_neural_style/images/content-images/soccer_ball.jpeg*

In order to create the mosaic version navigate to ***fast_neural_style*** directory and run:
```
python neural_style/neural_style.py 
       eval 
       --content-image images/content-images/soccer_ball.jpeg 
       --model saved_models/mosaic.pth 
       --output-image images/output-images/mosaic_soccer_ball_2.jpeg  
       --cuda 0 
       --mac-m1 1
```

The output image is stored in *fast_neural_style/images/output-images/mosaic_soccer_ball_2.jpeg*.

## New Dataset 

For the purpose of this task, the 320px version of [Imagenette](https://github.com/fastai/imagenette) is used. A subset of 100 images is used and 10 epochs selected to train the model. 

The images are randomly picked during training time and a new argument(***--subset***) is created in order to select the size of the subset(default: 100).

**The subset should be picked during training time and not as a preprossing step.** 

The interpretation of this sentence could have 2 meanings. The preprocessing step could either mean manually selecting a subset of the images and putting them in a folder or it could include the part of loading the dataset and parsing it to the DataLoader.  If the former is true then the subset could be selected by adding the following lines of code before the DataLoader:

```
train_dataset = datasets.ImageFolder(args.dataset, transform)
train_dataset_size = len(train_dataset)
random_ind = random.sample(range(0,train_dataset_size - 1), args.subset)
train_subset = torch.utils.data.Subset(train_dataset, random_ind)

train_loader = DataLoader(train_subset, batch_size=args.batch_size)
```

For this task we consider the latter to be true and the implementation is described below.


### Train 
Since the part before the DataLoader is considered as preprocessing step we have to pick the subset inside each epoch. 

A new data loader similar to the ImageFolder is implemented. The main difference is that the samples containing the paths of the images are shuffled. Therefore, during each epoch we can pick from the start batches of data until we reach the desired size of the subset and the result would be a subset of images from many classes(Imagenette has a total of 10 classes). If we use the usual ImageFolder then the model would be trained by images of the same class.

***DataLoader* is implemented with shuffle=False, which guarantees that the batches will have the same order during each epoch.**

Inside ***fast_neural_style*** directory run:
```
python neural_style/neural_style.py 
       train 
       --dataset images/imagenette2-320 
       --image-size 320 
       --style-image images/style-images/mosaic.jpg 
       --save-model-dir saved_models/ 
       --epochs 10 
       --cuda 0 
       --mac-m1 1
```
### Style
By executing the code as in task 1 with the new model and specifying a new name for the output image we get the image stored in  *fast_neural_style/images/output-images/mosaic_soccer_ball_3.jpeg*

## Data Augmentation

During this task we add **RandomRotation** as data augmentation and the relevant parameters as command line arguments. The command line arguments are ***--rotation-degrees*** and ***--rotation-fill***. Both support a single number or a sequence of two numbers separated by comma and both have a default value of 0. 

<a name="train1"></a>
### Train 

Inside ***fast_neural_style*** directory run:
```
python neural_style/neural_style.py 
       train 
       --dataset images/imagenette2-320 
       --image-size 320 
       --style-image images/style-images/mosaic.jpg 
       --save-model-dir saved_models/  
       --rotation-fill 0 
       --epochs 10 
       --cuda 0 
       --mac-m1 1 
       --rotation-degrees 50,50
```

<a name="style1"></a>
### Style
Similarly as before, we get the image stored in *fast_neural_style/images/output-images/mosaic_soccer_ball_4.jpeg*

## Activation function
The purpose of this task is to change the activation function in the residual block of the transformer network from **ReLU** to **RReLU**.

> **_NOTE:_** The operator 'aten::rrelu_with_noise' is not current implemented for the MPS device(MacOS). As a consequence, the model was trained and the style was done using CPU.

<a name="train2"></a>
### Train

Inside ***fast_neural_style*** directory run:
```
python neural_style/neural_style.py 
       train 
       --dataset images/imagenette2-320 
       --image-size 320 
       --style-image images/style-images/mosaic.jpg 
       --save-model-dir saved_models/  
       --rotation-fill 0 
       --epochs 10 
       --cuda 0 
       --mac-m1 0 
       --rotation-degrees 25
```

<a name="style2"></a>
### Style
Using the new model we get the image stored in *fast_neural_style/images/output-images/mosaic_soccer_ball_5.jpeg*