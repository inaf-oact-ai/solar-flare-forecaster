# solar-flare-forecaster
Solar flare forecaster application based on transformer models

## **About**
This repository implements a solar flare forecasting application with modern transformer backbones across three modalities — images, videos, and time series — using a single, consistent training/evaluation framework:   

* **Image forecaster (SigLIP2)** — processes single SDO/HMI magnetogram crops via a ViT encoder; head is a lightweight classifier. Trained with class-rebalancing strategies and weighted losses.
* **Video forecaster (VideoMAE)** — processes short sequences (e.g., 16 frames) of magnetograms with a spatio-temporal ViT backbone; head is a classifier on the pooled representation.
* **Time-series forecaster (Moirai2)** — processes GOES XRS flux-ratio sequences (and an optional flare-history channel) with a transformer encoder; head is a classifier over the forecasting target.

This documentation provides ready-to-use recipes plus links to trained checkpoints.  

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

## **Usage**  

To run the forecaster application on input data use the the provided script ```run.py```:   

```(sfforecaster)$ python $VENV_DIR/sfforecaster/bin/run.py [OPTIONS]```    

Supported options are: 

**INPUT DATA**  
`--datalist=[VALUE]`: Path to input training/test data in json format   
`--datalist_cv=[VALUE]`: Path to input validation data in json format   
`--ts_vars=[VALUE]`: Name of time series variables in input json data, separated by commas. Default: 'xrs_flux_ratio,flare_hist'        
`--ts_npoints=[VALUE]`: Number of points in time series variables. Default: 1440    

For HMI image inputs, the required json format is:   

```json
{
  "data": [
    {
      "filepath": "PATH-TO-HMI-IMAGE",
      "label": "M+",
      "id": 1,
      "flare_type": "X",
      "flare_id": 3
    },
    {
      ...
      ...
    }
  ]
}
```

For HMI video inputs, the required json format is:   

```json
{
  "data": [
    {
      "filepaths": ["PATH-TO-FIRST-HMI-FRAME", ..., "PATH-TO-LAST-HMI-FRAME"],
      "label": "M+",
      "id": 1,
      "flare_type": "X",
      "flare_id": 3
    },
    {
      ...
      ...
    }
  ]
}
```

For time series inputs, the required json format is:   

```json
{
  "data": [
    {
      "tsvar1": ["PATH-TO-FIRST-TS-ENTRY", ..., "PATH-TO-LAST-TS-ENTRY"],
      "tsvar2": ["PATH-TO-FIRST-TS-ENTRY", ..., "PATH-TO-LAST-TS-ENTRY"],
      ...
      ...
      "tsvarN": ["PATH-TO-FIRST-TS-ENTRY", ..., "PATH-TO-LAST-TS-ENTRY"],
      "label": "M+",
      "id": 1,
      "flare_type": "X",
      "flare_id": 3
    },
    {
      ...
      ...
    }
  ]
}
```
In this case, you should specify the name of time series variables in the `--ts_vars` option, e.g. `--ts_vars="tsvar1,tsvar2,...,tsvarN"`.     

**DATA PRE-PROCESSING (IMAGES/VIDEOS)**     
`--zscale`: Apply zscale transform to input images. Default: not applied.   
`--zscale_contrast=[VALUE]`: zscale contrast parameter. Default: 0.25   
`--grayscale`: Load input images in grayscale (1 chan tensor). Default: load as 3-chan RGB     
`--use_model_processor`: Transform data using available model image processor in data collator. Default: not used         
`--resize`: Resize input image before model processor (if enabled). Default: no resize      
`--resize_size=[VALUE]`: Resize size in pixels used if --resize option is enabled. Default: 224        
`--asinh_stretch`: Apply asinh stretch transform to input images. Default: not applied   
`--pmin=[VALUE]`: Min percentile parameter for asinh transform. Default: 0.5   
`--pmax=[VALUE]`: Max percentile parameter for asinh transform. Default: 99.5   
`--asinh_scale=[VALUE]`: asinh_scale parameter for asinh transform . Default: 0.5   

**DATA PRE-PROCESSING (TIME-SERIES)**    
`--ts_logstretchs=[VALUE]`: Log stretch TS vars separated by commas (1=enable, 0=disable). Must have same dimension of ts_vars. Default: "0,0"       	
 
**DATA AUGMENTATION (IMAGES/VIDEOS)**      
`--add_crop_augm`: If enabled, add random center crop and resize (--resize_size) augmentation in training. Default: not applied.      
`--min_crop_fract=[VALUE]`: Mininum crop fraction. Default: 0.65             	 

**MODEL (COMMON)**      
`--data_modality=[VALUE]`: Data modality model used: {"image","video","ts"}. Default: "image"      
`--model=[VALUE]`: Model pretrained file name or weight path to be loaded {"google/siglip-so400m-patch14-384","MCG-NJU/videomae-base"}. Default: "google/siglip-so400m-patch14-384"     
`--binary`: Set binary classification label scheme. Default: multiclass   
`--flare_thr=[VALUE]`: Choose flare class label name: {C-->label=C+,M-->label=M+}. Default: M      
`--binary_thr=[VALUE]`: Binary decision threshold (used in eval/test). Default: 0.5   

**MODEL (IMAGES/VIDEOS)**  
`--vitloader`: If enabled use ViTForImageClassification to load image model otherwise AutoModelForImageClassification. Default: disabled        
`--video_model`: Video model used: {"videomae","imgfeatts"}. Default: "videomae"           
`--freeze_backbone`: Make image/video encoder backbone layers non-trainable. Default: all trainable        
`--max_freeze_layer_id`: ID of the last layer kept frozen. -1 means all are frozen if --freeze_backbone option is enabled. Default: -1          	

**MODEL (TIME SERIES)**  
`--model_ts_backbone=[VALUE]`: Time series model backbone name. Default: "Salesforce/moirai-2.0-R-small"         
`--model_ts_img_backbone=[VALUE]`: Image backbone model used in "imgfeatts" model type. Default: "google/siglip2-base-patch16-224"             
`--ts_patching_mode=[VALUE]`: Patching mode used in time series model with input ts variates: {"time_only","time_variate"}. Default: "time_variate"              
`--proj_dim=[VALUE]`: Size of linear projection layer in ImageFeatTSClassifier model. Default: 128               
`--ts_freeze_backbone`: Make Moirai backbone layers non-trainable. Default: all trainable               
`--ts_max_freeze_layer_id`: ID of the last layer kept frozen. -1 means all are frozen if --ts_freeze_backbone option is enabled. Default: -1          	

**MODEL TRAINING**  
`--run_eval_on_start`: Run model evaluation on start for debug. Default: disabled    
`--run_eval_on_start_manual`: Run model evaluation manually on start for debug. Default: disabled    
`--run_eval_on_step=[VALUE]`: Run model evaluation after each step. Default: disabled    
`--logging_steps=[VALUE]`: Number of logging steps. Default: 1    
`--gradient_accumulation_steps=[VALUE]`: Number of updates steps to accumulate the gradients for, before performing a backward/update pass. Default: 1    
`--nepochs=[VALUE]`: Number of epochs used in network training. Default: 1    
`--lr_scheduler=[VALUE]`: Learning rate scheduler used: {"constant", "linear", "cosine", "cosine_with_min_lr"}. Default: "cosine"       
`--lr=[VALUE]`: Learning rate used. Default: 5e-5        
`--warmup_ratio=[VALUE]`: Warmup ratio parameter used. Default: 0.2                
`--batch_size=[VALUE]`: Batch size used in training. Default: 8               
`--batch_size_eval=[VALUE]`: Batch size used for evaluation. If None set equal to train batch size. Default: None                 
`--drop_last`: Drop last incomplete batch. Default: disabled                  
`--weight_decay=[VALUE]`: AdamW weight decay. Default: 0.0                
`--head_dropout=[VALUE]`: Dropout prob before classifier heads. Default: 0.0                  
`--proj_dropout=[VALUE]`: Dropout prob applied to per-timestep projected features before Moirai (imgfeatts model). Default: 0.0               
`--ddp_find_unused_parameters`: Flag passed to DistributedDataParallel when using distributed training. Default: disabled               

	

	
	
	
