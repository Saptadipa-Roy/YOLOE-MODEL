# **YOLOE Object Detection**



##### **1. Project Overview**



This project explores object detection and instance segmentation using pretrained YOLOE models from Ultralytics. The objective is to test pretrained models on images and videos and experiment with prompt-free and text-prompt-based detection.



The project focuses on running inference, observing detection results, and evaluating model behavior on different objects and real-world video footage.



**Note:** This project currently uses pretrained models. Custom model training has not been performed as part of this work.



##### **2. Technologies Used**



\- Python

\- Ultralytics YOLOE

\- Roboflow (where applicable in the workflow)



##### **3. Project Features**



\- Testing pretrained YOLOE models

\- Image-based object detection

\- Video inference

\- Prompt-free object detection

\- Text-prompt-based object detection

\- Visualization of detection results and bounding boxes



##### **4. Models Tested**



\- `yoloe-11l-seg-pf.pt` — used for prompt-free inference

\- `yoloe-11l-seg.pt` — used for text-prompt-based inference



##### **5. Installation**



###### **Prerequisites**



\- Python

\- A compatible environment for running Ultralytics YOLOE

\- Sufficient system resources for image and video inference



Clone the repository:





git clone https://github.com/Saptadipa-Roy/YOLOE-MODEL.git

cd YOLOE-MODEL





Create and activate a virtual environment on Windows:





python -m venv .venv

.venv\\Scripts\\activate





Install Ultralytics:





python -m pip install --upgrade pip

python -m pip install ultralytics





Additional dependencies may be required for particular model features, especially text-prompt-based inference.



##### **6. Running Inference**



###### **A. Prompt-Free Inference**



The following example runs a pretrained YOLOE model on a video without specifying a text prompt.





from ultralytics import YOLOE



model = YOLOE("yoloe-11l-seg-pf.pt")



results = model.predict(

&#x20;   source="Room-Video.mp4",

&#x20;   show=True,

&#x20;   save=True,

&#x20;   show\_conf=False

)





Replace `Room-Video.mp4` with the path to your input image or video.



###### **B. Text-Prompt-Based Inference**



This example uses a text prompt to guide detection.





from ultralytics import YOLOE



model = YOLOE("yoloe-11l-seg.pt")



names = \["camel walking"]

model.set\_classes(names, model.get\_text\_pe(names))



results = model.predict(

&#x20;   source="Camel Walking.mp4",

&#x20;   show=True,

&#x20;   save=True

)





Replace `"camel walking"` with the object or concept you want to test and provide the appropriate input file.



**Note:** Text-prompt-based inference may require additional dependencies, depending on the installed Ultralytics version.



##### **7. Testing and Observations**



The pretrained models were tested on different images and video footage to observe their detection behavior.



The testing included:

\- Checking whether objects were detected in images.

\- Observing bounding boxes around detected objects.

\- Running inference on indoor video footage.

\- Experimenting with prompt-free and text-prompt-based inference.



Detection results may vary depending on the input, model configuration, and object being tested.



##### **8. Using a Trained Model**



This project currently uses pretrained checkpoints and does not include a custom-trained model.



If custom training is performed in the future, the resulting checkpoint can be loaded for inference using a compatible YOLOE model configuration. The training workflow should be verified against the relevant Ultralytics documentation and model version.



##### **9. Limitations**



\- Inference speed depends on the hardware and model size.

\- Large models may run slowly on systems with limited computing resources.

\- Detection results depend on the input quality and model capabilities.

\- Text-prompt-based inference may require additional dependencies.

\- No custom dataset training or fine-tuning has been performed in this project.



##### **10. References**



\- Ultralytics documentation: https://docs.ultralytics.com/

\- Ultralytics GitHub repository: https://github.com/ultralytics/ultralytics

\- Project repository: https://github.com/Saptadipa-Roy/YOLOE-MODEL



