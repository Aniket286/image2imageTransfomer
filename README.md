# image2imageTransfomer
Transformer Model that can be trained to image2image tranformation

# Requirements:
The below list libraries are:
OpenCV: opencv-python
NumPy: numpy
OS: Included in the Python Standard Library
Torch: torch
TensorboardX: tensorboardX
Torchvision: torchvision
YAML: pyyaml
Requests: requests
BeautifulSoup: beautifulsoup4

This can be installed by running:
pip install opencv-python numpy torch tensorboardX torchvision pyyaml requests beautifulsoup4

# Contour Generation:
Contours for test images can be generated running contour.py specifying the path of source and destination folder.

# For Training: 
The config file with training parameters and training data should be made as given example in config folder
Following command is run: python train.py --config /path/to/config.pt
