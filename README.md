# CT Skull Fractures
A skull fracture is a break in one or more of the eight bones that form the cranial portion of the skull, usually occurring as a result of blunt force trauma. If the force of the impact is excessive, the bone may fracture at or near the site of the impact and cause damage to the underlying structures within the skull such as the membranes, blood vessels, and brain. [wikipedia](https://en.wikipedia.org/wiki/Skull_fracture)

## pytorch & densenet161
using deep convolutional networks to improve fractures prediction in skull CT scans

## Recommended:
* python=3.6
* torch=1.5.1
* torchvision=0.5.0
* sklearn=0.20.1
* PIL=7.2.0

## Initialize:
```
mkdir plots/
mkdir model_outputs/
cd interactive
mkdir plots/
mkdir model_outputs/
```

#### Directory Structure:
```
CT_Skull_Fractures
└──├── plots(folder)
   ├── model_outputs(folder)
   ├── train.py
   ├── predict.py
   ├── README.md
   └── interactive(folder)
   |   ├── plots(folder)
   |   ├── model_outputs(folder)
   |   ├── train.ipynb
   |   └── predict.ipynb
   └── data(folder)
       ├── train(folder)
       |   ├── b(folder)
       |   |   ├── 105.png
       |   |   ├── 106.png
       |   ├── nb(folder)
       |       ├── 205.png
       |       ├── 206.png       
       ├── test(folder)
           ├── b(folder)
           |   ├── 1105.png
           |   ├── 1106.png           
           ├── nb(folder)    
               ├── 2105.png
               ├── 2106.png
```       
    
## Dataset:
The dataset consists of 8.685 png images of actual brain CT scans, of which 1.532 show some
form of broken skull (fracture or craniotomy) while the other 7.153 images do not. The images are
grayscale png format, have been zoomed into the brain and cropped to 340x340 pixels. The
images are a subset of the [RSNA-ASNR Intracranial Hemorrhage Detection Challenge image datasets and annotation files](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
* Main task: Classification

## Training:
* [train.py](https://github.com/jonykoren/CT_Skull_Fractures/blob/master/train.py)
```
python train.py
```

## Predict:
* [predict.py](https://github.com/jonykoren/CT_Skull_Fractures/blob/master/predict.py)
```
python predict.py
```

## Interactive 
#### Note: you can start by small training with the interactive notebooks:
* [train.ipynb](https://github.com/jonykoren/CT_Skull_Fractures/blob/master/interactive/train.ipynb)
* [predict.ipynb](https://github.com/jonykoren/CT_Skull_Fractures/blob/master/interactive/predict.ipynb)

<p align="center">
  <img src="https://github.com/jonykoren/CT_Skull_Fractures/blob/master/img/gif.gif?raw=true">
</p>

### <p align="center">feel free to ⭐️</p>

## Author
`Maintainer` [Jonatan Koren](https://jonykoren.github.io/) (jonykoren@gmail.com)
