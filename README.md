# Deploy a People Counter App at the Edge

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |

![people-counter-python](./images/people-counter-image.png)

## What it Does

The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app detects people in a designated area, provides the number of people in the frame, average duration of people in frame, and total count.

## How it Works

The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. I used the YOLO object detector to detect people in both a images and video streams with OpenCV and Python. 

The app does the following:

* Count the number of people in the current frame
* Calculate the duration that a person is in the frame (time elapsed between entering and exiting a frame)
* Count the total number of people occurred 
* Sends the data to a local web server using the Paho MQTT Python package

You'll need to convert the pre-trained YOLO model to IR for this to work.

![architectural diagram](./images/arch_diagram.png)

## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)
* OR Udacity classroom workspace for the related course

### Software

* Intel® Distribution of OpenVINO™ toolkit 2019 R3 release

* Node v6.17.1

* Npm v3.10.10

* CMake

*   MQTT Mosca server
  
*   TensorFlow 1.12
  
    ​    
## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)

### Install Nodejs and its dependencies

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)

### Install npm

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server 
-   Node.js* Web server
-   FFmpeg server
    

From the main directory:

* For MQTT/Mosca server:
   ```
   cd webservice/server
   npm install
   ```

* For Web server:
  ```
  cd ../ui
  npm install
  ```
  **Note:** If any configuration errors occur in mosca server or Web server while using **npm install**, use the below commands:
   ```
   sudo npm install npm -g 
   rm -rf node_modules
   npm cache clean
   npm config set registry "http://registry.npmjs.org"
   npm install
   ```

### Convert Models to the Intermediate Representation (IR)

This app uses a TensorFlow implementation of YOLOv3-tiny. 

Clone the repository: https://github.com/mystic123/tensorflow-yolo-v3

```
git clone https://github.com/mystic123/tensorflow-yolo-v3
d tensorflow-yolo-v3
git checkout ed60b90
```

Download [coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names) file from the DarkNet website

```
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

Download [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights) (for the YOLOv3-tiny model). 

```
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

Note you can also run the app with [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) or your own pre-trained weights with the same structure.

Generate the pre-trained TensorFlow model:

```
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
```

Create directory to save IR files. FP32 is for CPU. FP16 is for GPU and VPU.

```
mkdir -p FP32
mkdir -p FP16
```

Convert YOLOv3-tiny TensorFlow Model to the IR. 

```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --batch 1 --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json -o FP32

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --batch 1 --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json -o FP16 --data_type FP16
```

## Run the application

From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at: 

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```

*Depending on whether you are using Linux or Mac, the filename will be either `libcpu_extension_sse4.so` or `libcpu_extension.dylib`, respectively.* (The Linux filename may be different if you are using a AVX architecture)

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-yolov3-tiny-model.xml -d CPU -pt 0.01 -iout 0.0001 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/tensorflow-yolo-v3/FP32_tiny/frozen_darknet_yolov3_model.xml -d CPU -pt 0.01 -iout 0.0001 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 10 -i - http://0.0.0.0:3004/fac.ffm
```
If you are in the classroom workspace, use the “Open App” button to view the output. If working locally, to see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

#### Running on the Intel® Neural Compute Stick

To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

```
python3.5 main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-yolov3-tiny-model.xml -pt 0.01 -iout 0.0001 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

**Note:** The Intel® Neural Compute Stick can only run FP16 models at this time. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

#### Using a camera stream instead of a video file

To get the input video from the camera, use the `-i CAM` command-line argument. Specify the resolution of the camera using the `-video_size` command line argument.

For example:
```
python main.py -i CAM -m your-yolov3-tiny-model.xml -d CPU -pt -pt 0.01 -iout 0.0001 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

**Note:**
User has to give `-video_size` command line argument according to the input as it is used to specify the resolution of the video or image file.

## Explaining Custom Layers

OpenVINO supports various common neural network layers in frameworks like TensorFlow and Caffe. Find the list of the supported layers it supports [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html). 

Although the list is extensive, you might still work with models with layers not on this list. They are called *custom layers*. You will need to add extensions to both *Model Optimizer* and *Inference Engine* to use custom layers. 

The Model Optimizer coverts the input model to IR format, and the Inference Engine runs inference on the IR model at the edge. Both use OpenVINO's built-in libraries to do their tasks with the supported layers. However, to work with custom layers, you need to add extensions to these libraries.

The process behind converting custom layers involves following steps:

![MEG_generic_flow.png](https://docs.openvinotoolkit.org/latest/MEG_generic_flow.png)

1. **Generate**: Use the Model Extension Generator to generate the Custom Layer Template Files
2. **Edit**: Edit the Custom Layer Template Files as necessary to create the specialized Custom Layer Extension Source Code
3. **Specify**: Specify the custom layer extension locations to be used by the Model Optimizer or Inference Engine.

For details, refer to OpenVINO's [Custom Layers Guide](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html). 

Note this project doesn't use custom layers.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were...

|                | Method                                                  | Before    | After                          |
| -------------- | ------------------------------------------------------- | --------- | ------------------------------ |
| Accuracy       | Run inference on an image with 21 people                | 17 people | 11 people                      |
| Size           | Compare frozen model (pb) size with IR model (bin) size | 34,649KB  | FP32 34,568KB. FP16FP 17,284KB |
| Inference time | Run inference on an image                               | 2.12 s    | 2.485 ms                       |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

1. Retail stores. We can use this app to get a store's conversion rate - how many people who entered the store actually made the purchase. This might be useful. The store can run experiments on their business process, and use the conversion rate as a metrics to measure the performance.
2. Meeting rooms. We can use this app to understand how the rooms are used. This might be useful. If a meeting room is rarely used, or it is often over-crowded, the company might want to investigate on the reasons behind. They might as well consider to re-design their office spaces.
3. Gym. We can use this app to observe the crowdedness in different time period (day, hours). This might be useful. The gym can automatically optimize the settings of its air conditioning and lighting to save energy.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are:

* Lighting. Good lighting is important for this model to work. If the app has to work in dark environment, we might need to apply additional pre-processing to the images, converting dark images to normal lighting images. We might also make this work by training another model with low-light dataset.
* Model accuracy. If a better accuracy is required, we might want to use SSD over YOLO. YOLO is super fast, but not as accurate as SSD. 
* Camera focal length/image size. If we need to know the distance or the physical size of each person, the app will need to know the camera's focal length. Otherwise, it is irrelevant. And, every image gets pre-processed (resized) before it goes into the model. So, the camera's image size is not of our concerns.

## Future Works

1. Add an alarm or notification when the app detects above a certain number of people on video, or people are on camera longer than a certain length of time.
2. Try out different models than the People Counter, including a model you have trained. Note that this may require some alterations to what information is passed through MQTT and what would need to be displayed by the UI.
3. Deploy to an IoT device (outside the classroom workspace or your personal computer), such as a Raspberry Pi with Intel® Neural Compute Stick.
4. Add a recognition aspect to your app to be able to tell if a previously counted person returns to the frame. The recognition model could also be processed on a second piece of hardware.
5. Add a toggle to the UI to shut off the camera feed and show stats only (as well as to toggle the camera feed back on). Show how this affects performance (including network effects) and power.