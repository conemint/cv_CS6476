These are starter pack for final project code for CS6476 stereo correspondence.

input images:
    sourcs: http://vision.middlebury.edu/stereo/data/scenes2014/
    Quarter resolution images of left, right, Ground truth for left view, Ground truth for right view.
    folder structure:
    input_images/
    ├── groundTruthLeft
    ├── groundTruthRight
    └── trainingQ

A list of code files will be included.
    
 - pfmreader.py: read and process ground truth files in .pfm format. Output np.array of depths and grayscale .png for display purposes.

 - stereo.py: stereo class.
 
 - experiment.py: