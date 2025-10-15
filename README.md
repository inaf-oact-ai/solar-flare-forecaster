# solar-flare-forecaster
Solar flare forecaster application based on transformer models

## **About**
Forecast solar flares with modern transformer backbones across three modalities — images, videos, and time series — using a single, consistent training/evaluation framework. The code accompanies the paper "Solar flare forecasting with foundational transformer models across image, video, and time-series modalities” (Riggi et al., 2025) and provides ready-to-use recipes plus trained checkpoints.

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add a reference to this github repository and acknowledge these works in your paper:   

* S. Riggi et al., *Solar flare forecasting with foundational transformer models across image, video, and time-series modalities*, 2025

## **Installation**  

To build and install the package, we recommend Python ≥ 3.10 and a recent CUDA-enabled PyTorch:    

* Download the software in a local directory, e.g. ```SRC_DIR```:   
  ```$ git clone https://github.com/inaf-oact-ai/solar-flare-forecaster.git```
  
* Create and activate a virtual environment, e.g. ```sfforecaster```, under a desired path ```VENV_DIR```     
  ```$ python3 -m venv $VENV_DIR/sfforecaster```    
  ```$ source $VENV_DIR/sfforecaster/bin/activate```
  
* Install dependencies inside venv:   
  ```(sfforecaster)$ pip install -r $SRC_DIR/requirements.txt```
  
* If you intend to use Moirai2 time series model, you need to download the latest `uni2ts` module (not tagged moirai v1 versions) in a desidered path (e.g. ```$MOIRAI_SRC_DIR```) and install it:      
  ```(sfforecaster)$ mkdir $MOIRAI_SRC_DIR```
  ```(sfforecaster)$ cd $MOIRAI_SRC_DIR```    
  ```(sfforecaster)$ git clone https://github.com/SalesforceAIResearch/uni2ts.git```    
  ```(sfforecaster)$ cd uni2ts```     
  ```(sfforecaster)$ pip install -e '.[notebook]'```

  **NB: uni2ts install can override your previously installed torch and torchvision version. In case of conflicts, manually adjust and align versions.**
  
* Build and install package in virtual env:    
  ```(sfforecaster)$ cd $SRC_DIR```    
  ```(sfforecaster)$ python setup.py install```    
       
To use package scripts:

* Add binary directory to your ```PATH``` environment variable:   
  ``` export PATH=$PATH:$VENV_DIR/sfforecaster/bin ```    
