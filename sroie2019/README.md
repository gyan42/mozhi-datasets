# Receipts Dataset

Official URL: https://rrc.cvc.uab.es/?ch=13&com=downloads
Drive URL: https://drive.google.com/open?id=1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2
Drive 2 Link: https://drive.google.com/file/d/1cD8Y_qzSaZPjMskFIwI7M6TIEz1P59Pd/view?usp=sharing


Tasks - ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction
Dataset and Annotations

The dataset will have 1000 whole scanned receipt images. Each receipt image contains around about four key text fields, such as goods name, unit price and total cost, etc. The text annotated in the dataset mainly consists of digits and English characters. An example scanned receipt is shown below:

sample21.jpg

 

The dataset is split into a training/validation set (“trainval”) and a test set (“test”). The “trainval” set consists of 600 receipt images which will be made available to the participants along with their annotations. The “test” set consists of 400 images, which will be made available a few weeks before the submission deadline.

For receipt OCR task, each image in the dataset is annotated with text bounding boxes (bbox) and the transcript of each text bbox. Locations are annotated as rectangles with four vertices, which are in clockwise order starting from the top. Annotations for an image are stored in a text file with the same file name. The annotation format is similar to that of ICDAR2015 dataset, which is shown below:

x1_1, y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript_1

x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2, transcript_2

x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3, transcript_3

…

For the information extraction task, each image in the dataset is annotated with a text file with format shown below:

{"company": "STARBUCKS STORE #10208",

"date": "14/03/2015",

"address": "11302 EUCLID AVENUE, CLEVELAND, OH (216) 229-0749",

"total": "4.95",
}

 
Task 1 - Scanned Receipt Text Localisation
Task Description

Localizing and recognizing text are conventional tasks that has appeared in many previous competitions, such as the ICDAR Robust Reading Competition (RRC) 2013, ICDAR RRC 2015 and ICDAR RRC 2017 [1][2]. The aim of this task is to accurately localize texts with 4 vertices. The text localization ground truth will be at least at the level of words. Participants will be asked to submit a zip file containing results for all test images.
Evaluation Protocol

As participating teams may apply localization algorithms to locate text at different levels (e.g. text lines), for the evaluation of text localizazation in this task, the methodology based on DetVal will be implemented. The methodology address partly the problem of one-to-many and many to one correspondences of detected texts. In our evaluation protocol mean average precision (mAP) and average recall will be calculated, based on which F1 score will be computed and used for ranking [3].

 
Task 2 - Scanned Receipt OCR
Task Description

The aim of this task is to accurately recognize the text in a receipt image. No localisation information is provided, or is required. Instead the participants are required to offer a list of words recognised in the image. The task will be restricted to words comprising Latin characters and numbers only.

The ground truth needed to train for this task is the list of words that appear in the transcriptions. In order to obtain the ground truth for this task, you should tokenise all strings splitting on space. For example the string “Date: 12/3/56” should be tokenised “Date:”, “12/3/56”. While the string “Date: 12 / 3 / 56” should be tokenised “Date:” “12”, “/”, “3”, “/”, “56”.
Evaluation Protocol

For the Recognition ranking we will match all words provided by the participants to the words in the ground truth. If certain words are repeated in an image, they are expected to also be repeated in the submission results. We will calculate Precision (# or correct matches over the number of detected words) and Recall (# of correct matches over the number of ground truth words) metrics, and will use the F1 score as the final ranking metric.

 
Task 3 - Key Information Extraction from Scanned Receipts
Task Description

The aim of this task is to extract texts of a number of key fields from given receipts, and save the texts for each receipt image in a json file with format shown in Figure 3. Participants will be asked to submit a zip file containing results for all test invoice images.
Evaluation Protocol

For each test receipt image, the extracted text is compared to the ground truth. An extract text is marked as correct if both submitted content and category of the extracted text matches the groundtruth; Otherwise, marked as incorrect. The precision is computed over all the extracted texts of all the test receipt images. F1 score is computed based on precision and recall. F1 score will be used for ranking. 

