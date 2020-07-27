# CT_Skull_Fractures
A skull fracture is a break in one or more of the eight bones that form the cranial portion of the skull, usually occurring as a result of blunt force trauma. If the force of the impact is excessive, the bone may fracture at or near the site of the impact and cause damage to the underlying structures within the skull such as the membranes, blood vessels, and brain. [wikipedia](https://en.wikipedia.org/wiki/Skull_fracture)

## pytorch & densenet161
using deep convolutional networks to improve fractures prediction in skull CT scans

## Recommended:
* python=3.6
* torch=1.5.1
* torchvision=0.5.0
* sklearn=0.20.1
* PIL=7.2.0

## Initalize Directories:
```
mkdir plots/
mkdir model_outputs/
cd interactive
mkdir plots/
mkdir model_outputs/
```

#### Directory Structure
```
CT_Skull_Fractures
└──├ data(folder)
   ├── plots(folder)
   ├── model_outputs(folder)
   ├── train.py
   ├── predict.py
   ├── README.md
   └── interactive(folder)
       ├── plots(folder)
       ├── model_outputs(folder)
       ├── train.ipynb
       └── predict.ipynb
```       
    

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


<p align="center">
  <img src="https://github.com/jonykoren/CT_Skull_Fractures/blob/master/img/gif.gif?raw=true">
</p>

### <p align="center">feel free to ⭐️</p>

## Author
`Maintainer` [Jonatan Koren](https://jonykoren.github.io/) (jonykoren@gmail.com)
