# Darknet yolov4-tiny (feat. Tensorflow Lite)

The ```$root``` path is assumpted as current repository path. <br/>
This repository includes a setupt for trianing person detector, but you can apply this method for training the other dataset.</br>



## Compile Darknet 
### 1. Clone ['darknet' git repository](https://github.com/AlexeyAB/darknet) to the current ```$root```. 
```bash
~$ git clone https://github.com/AlexeyAB/darknet 
```

### 2. Change ```makefile``` to have GPU and OPENCV enabled # also set CUDNN, CUDNN_HALF and LIBSO to 1
```bash
~$ cd ./darknet/

~$ sed -i 's/OPENCV=0/OPENCV=1/' Makefile
~$ sed -i 's/GPU=0/GPU=1/' Makefile
~$ sed -i 's/CUDNN=0/CUDNN=1/' Makefile
~$ sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
~$ sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```

### 3. Build darknet 
```bash
~$ sudo apt install libopencv-dev   # (ref) https://github.com/pjreddie/darknet/issues/2280

~$ make 
```

### 4. Clean the data and cfg folders first except the labels folder in data which is required
``` bash 
~$ cd ./darknet/data 
~$ find -maxdepth 1 -type f -exec rm -rf {} \;

~$ cd .. 
~$ rm -rf cfg/ 
~$ mkdir cfg 
```

### 5. Unzip the datasets and their contents so that they are now in ```/darknet/data/``` folder
* you can preprare your own customized dataset 
* if you want to know how to prepare custom dataset, refer to [this article](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593#a70f).
``` bash 
~$ unzip ./obj.zip -d ./darknet/data 
```

### 6. Copy the custom cfg file from the drive to the darknet/cfg folder
```bash
~$ cp ./yolov4-tiny-custom.cfg ./darknet/cfg 
```


### 7. Copy the obj.names and obj.data files so that they are now in ```/darknet/data/``` folder
```bash
~$ cp ./yolov4-tiny/obj.names ./darknet/cdata
~$ cp ./yolov4-tiny/obj.data  ./darknet/data
```

### 8. Copy the process.py file from the drive to the darknet directory
```bash
~$ cp ./process.py ./darknet/
```

### 9. Run the ```process.py``` python script to create the ```train.txt``` & ```test.txt``` files inside the data folder.
```bash
~$ cd ./darknet/ 

~$ python process.py      # this creates the train.txt and test.txt files in our darknet/data folder
~$ ls ./darknet/data/     # list the contents of data folder to check if the train.txt and test.txt files have been created 
```

### 10. Download the pre-trained ```yolov4-tiny``` weights.
```bash
~$ cd ./darknet/ 
~$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29  # Download the yolov4-tiny pre-trained weights file
```

***

## Training 
train your custom detector!
```bash 
~$ cd ./darknet/ 
~$ mkdir ./darknet/data/training     # train checkpoint will be saved here 

~$ ./darknet detector train ./data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map   # run training 

```
(option) to restart training your custom detector where you left off(using the weights that were saved last) (히스토리에서 가져올 때)
```bash
~$ ./darknet detector train ./data/obj.data cfg/yolov4-tiny-custom.cfg ./data/training/yolov4-tiny-custom_last.weights -dont_show -map  # re-train from the checkpoint 
```


## check performance 
You can check the mAP for all the saved weights to see which gives the best results ( xxxx here is the saved weight number like 4000, 5000 or 6000 snd so on )
```bash
~$ cd ./darknet 
~$ ./darknet detector map ./data/obj.data ./cfg/yolov4-tiny-custom.cfg ./data/training/yolov4-tiny-custom_best.weights -points 0
```

*** 
<br/>

# Convert ```Darknet``` to  ```TensorFlow```

### 1. Clone ```'Tensorflow light for yolov4'```
```bash 
~$ git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
~$ cd ./tensorflow-yolov4-tflite
~$ mkdir checkpoints
~$ pip install -r requirements.txt 
```


### 2. Convert the weights to TensorFlow's ```.pb``` representation
```bash
~$ cp ./darknet/data/obj.names ./tensorflow-yolov4-tflite/data/classes/ 
```

### 3. Change the labels from the default COCO to your own custom ones. 
```bash 
~$ sed -i "s/coco.names/obj.names/g" ./tensorflow-yolov4-tflite/core/config.py
```

### 4. Convert darknet weights to tensorflow
Convert to both a regular TensorFlow SavedModel and to ```TensorFlow Lite```. <br/>
For ```TensorFlow Lite```, we'll convert to a different TensorFlow SavedModel beforehand.
```bash
~$ cd ./tensorflow-yolov4-tflite

~$ python save_model.py \
  --weights ../darknet/data/training/yolov4-tiny-custom_best.weights \
  --output ./checkpoints/yolov4-tiny-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \
```

## Run demo tensorflow 
``` bash 
~$ pip install -U opencv-python==4.1.2.30
~$ python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --image ./data/girl.png --tiny True --output 'result.png' --score 0.7   # check the results 
```


# Convert the ```Darknet model ``` weights to ```TensorFlow Lite```
```bash
~$ cd ./tensorflow-yolov4-tflite

~$ python save_model.py \
  --weights ../darknet/data/training/yolov4-tiny-custom_best.weights \
  --output ./checkpoints/yolov4-tiny-416-tflite\
  --input_size 416 \
  --model yolov4 \
  --tiny \
  --framework tflite

```


```bash
# # SavedModel to convert to TFLite
~$ python convert_tflite.py --weights ./checkpoints/yolov4-tiny-416-tflite --output ./checkpoints/yolov4-tiny-416.tflite

```

```bash
# Run demo tensorflow 
~$ python detect.py --weights ./checkpoints/yolov4-tiny-416.tflite --size 416 --model yolov4 --image ./data/girl.png --framework tflite --tiny --score 0.2
```


# Convert the ```TensorFlow``` weights to ```TensorFlow Lite``` 
```bash
~$ cd ./tensorflow-yolov4-tflite 

~$ python convert_tflite.py --weights ./checkpoints/yolov4-tiny-416-tflite  --output ./checkpoints/yolov4-tiny-416.tflite
```


***
# Reference 
[1] [AVA-Dataset-Processing-for-Person-Detection](https://github.com/DoranLyong/AVA-Dataset-Processing-for-Person-Detection) / for training person detection dataset <br/>
[2] [yolov4-tiny-tflite-for-person-detection](https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection) / an example of person detector trained by Darknet <br/>
[3] [TRAIN A CUSTOM YOLOv4-tiny OBJECT DETECTOR USING GOOGLE COLAB](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593#a70f) / 
