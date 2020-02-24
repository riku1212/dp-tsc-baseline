# Deep Learning for Time Series Classification with Differential Privacy
Adapted from https://github.com/hfawaz/dl-4-tsc/ with tensorflow-privacy.

# Project Structure
```
.
|--  dp-tsc-baseline
|    |-- privacy
|    |-- classifiers
|    |-- utils 
|    |-- results
|
|-- archives
|-- results
```

# Code
To run the model, we use
```
python main.py [Archive Name] [Dataset Name] [Classifier Name] [Comments]
```
where Archive Name can be `mts_archive`, `UCRArchive_2018`, or any specified archive located in the `./archives` directory. Usually, `Comments` are written in the format `_itr_[number]` to indicate number of iteration this model has been run.

The resulting model and evaluation metrics will be stored in `results/[Classifier Name]/[Archive Name][Comment]`.

For example, running the command
```
python main.py UCRArchive_2019 Coffee fcn _itr_8
```
would mean storing in `./results/fcn/UCRArchive_2018_itr_8`.

# Classifiers
Currently, the available classfiers are:
 + `fcn`
 + `mlp`
 + `resnet`
 + `tlenet`
 + `mcnn`
 + `twiesn`
 + `encoder`
 + `mcdcnn`
 + `cnn`
 + `inception` 
 
We aim to include `lstm` in this set and incorporate differential privacy.