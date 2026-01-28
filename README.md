# Mini Project: Face Indexing and Retrieval (LTP + PP)

This project implements LTP for Face Recognition with some Pre-Processing done to the raw images for better results.The system is divided into 2 moduless:
-An offline Indexing Modeule which is responsible for indexing facial features from the raw images (AT&T Database of Faces).
-An online Indexing module which basically takes an image and tries to find the best match for it based on the offline indexation result.


## Dependencies: ##
Python 3 
 ## libraries: ##
    - opencv
    - numpy
    - pandas
    - openpyxl

## Methodology: ##
    The system relies on a specific pipeline
    - PreProcessing chain to ensure robustness against lighting changes.(Gamma Correction, DoG, Masking, Contrast Equalization.... You know ,the usual stuff)
    - Feature Extraction using LTP(cus LBP is kinda stupid)
    - Matching & Classification using KNN

## Usage Instructions ##
 ** Step 1: Offline Indexation **
 just run "python 3 Mini Project LTP+PP/Offilne Indexation.py"
 This step processes the raw images, saves the pre-processed versions to disk, calculates their LTP features, and stores them in a JSON database.Ensure your raw images are in a folder named Faces in the same directory as the script.

 **Step 2: Online Evaluation **
just run "python 3 Mini Project LTP+PP/Online_Indexation.py"
This step simulates a retrieval scenario. It takes specific test images (currently configured for images 8, 9, and 10 of specific subjects), treats them as "unknown" queries, and attempts to recognize them against the index.json database.
