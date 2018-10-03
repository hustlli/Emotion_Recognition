
//This file load video clips, detect face images therein and recognize the emotion
//Author: Liang Li
//


#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <algorithm>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv.hpp>
#include "facedetect-dll.h"
#pragma comment(lib,"libfacedetect.lib")

#include "loadbmp.h"  
#include "ffmpeg.h"
#include "lbp_hf.h"

using namespace std;
using namespace cv;

#define  feature_num 1110 //number of features of LBF-HF descriptor

void face_detection_recognition(Mat& img, double scale);
void emotion_recognition_LBP(IplImage* src, Rect select);
void emotion_recognition_HOG(IplImage *image2, Rect select);
void emotion_recognition_LBP_HOG(IplImage *image2, Rect select);

IplImage *image1;

//seven emotion categories:angry, fear, disgust, happy, neutral, sad, surprise
CvSVM svm_happy;
CvSVM svm_surprise;
CvSVM svm_angry;
CvSVM svm_sad;
CvSVM svm_disgust;
CvSVM svm_fear;
CvSVM svm_neutral;

CvRTrees rtree_happy;
CvRTrees rtree_surprise;
CvRTrees rtree_angry;
CvRTrees rtree_sad;
CvRTrees rtree_disgust;
CvRTrees rtree_fear;
CvRTrees rtree_neutral;

int happy_result = 0;
int surprise_result = 0;
int angry_result = 0;
int sad_result = 0;
int disgust_result = 0;
int fear_result = 0;
int neutral_result = 0;

int main( int argc, const char** argv )
{
	Mat image;
	CascadeClassifier cascade, nestedCascade;//create cascade clsssifier object, for face detection
	double scale = 1.3;  //resizing scale for the image

	// load pre-trained classifiers

	// HOG descriptor && SVM
	//svm_happy.load("D:\\SVM_HOG_Happy.xml");
	//svm_surprise.load("D:\\SVM_HOG_Surprise.xml");
	//svm_angry.load("D:\\SVM_HOG_Angry.xml");
	//svm_sad.load("D:\\SVM_HOG_Sad.xml");
	//svm_fear.load("D:\\SVM_HOG_Fear.xml");
	//svm_disgust.load("D:\\SVM_HOG_Disgust.xml");
	//svm_neutral.load("D:\\SVM_HOG_Neutral.xml");

	// LBP-HF descriptor && SVM
	svm_happy.load("video_happy.txt");
	svm_surprise.load("video_surprise.txt");
	svm_angry.load("video_angry.txt");
	svm_sad.load("video_sad.txt");
	svm_fear.load("video_fear.txt");
	svm_disgust.load("video_disgust.txt");
	svm_neutral.load("video_neutral.txt");

	// HOG descriptor && Random Forest
	//rtree_happy.load("RandomForset_HOG_Happy.xml");
	//rtree_surprise.load("RandomForset_HOG_Surprise.xml");
	//rtree_angry.load("RandomForset_HOG_Angry.xml");
	//rtree_sad.load("RandomForset_HOG_Sad.xml");
	//rtree_fear.load("RandomForset_HOG_Fear.xml");
	//rtree_disgust.load("RandomForset_HOG_Disgust.xml");
	//rtree_neutral.load("RandomForset_HOG_Neutral.xml");

	// LBP-HF+HOG descriptor && SVM
	//svm_happy.load("SVM_LBP_HOG_Happy.txt");
	//svm_surprise.load("SVM_LBP_HOG_Surprise.txt");
	//svm_angry.load("SVM_LBP_HOG_Angry.txt");
	//svm_sad.load("SVM_LBP_HOG_Sad.txt");
	//svm_fear.load("SVM_LBP_HOG_Fear.txt");
	//svm_disgust.load("SVM_LBP_HOG_Disgust.txt");
	//svm_neutral.load("SVM_LBP_HOG_Neutral.txt");

	// get video path information, which is stored in filelist_test.txt
    string file[600];               
	int line_i=0; 
    ifstream infile("filelist_test.txt",ios::in);
 
    while(!infile.eof())            
    {  
        getline(infile, file[line_i], '\n'); 
        line_i++;                    
    }

	FFMpegVideoHander ffmpeg; //ffmpeg is uesd for handling video files

	// Load video clips
	for(int index = 560; index < 570; index++)        
    {   
		// index: index of the videos to be uploaded

		string filePath = "D:\\test\\";
		string filename = filePath + file[index];
		char fileName[50];
		string txtType = ".txt";
		string txtName = filePath + file[index].substr(0,9) + txtType;
		
		ofstream myfile(txtName,ios::out);

		int j;
		for(j=0;j<filename.length();j++)
		{
			fileName[j] = filename[j];
		}
		fileName[j] = '\0';
		cout << "处理：" << fileName << endl; 

		ffmpeg.Load(fileName,"");

		while (ffmpeg.GetNextFrame())
		{
			// load each frame and recognize the emotion there in
			image1 = ffmpeg.GetCurrFrame(); 
			image = cvarrToMat(image1);
			
			// show the current frame
			cvNamedWindow("emotion",1);
			cvShowImage("emotion",image1); 
			char c=cvWaitKey(33);

			// recognition result initialization
			happy_result = 0;
			surprise_result = 0;
			angry_result = 0;
			sad_result = 0;
			disgust_result = 0;
			fear_result = 0;
			neutral_result = 0;

			// detect face image in the frame and recognize the emotion
			image = cvarrToMat(image1);
			face_detection_recognition(image,scale);

			ffmpeg.release_img(); 
				
		}

		// record the numbers of different prediction results
		cout<<endl;
		cout<<"happy  "<<happy_result<<endl;
		cout<<"surprise  "<<surprise_result<<endl;
		cout<<"sad  "<<sad_result<<endl;
		cout<<"fear  "<<fear_result<<endl;
		cout<<"disgust  "<<disgust_result<<endl;
		cout<<"angry  "<<angry_result<<endl;
		cout<<"neutral  "<<neutral_result<<endl;

		// final predcition of the whole video
		int a[7];
		a[0] = disgust_result;
		a[1] = surprise_result;
		a[2] = fear_result;
		a[3] = sad_result;
		a[4] = happy_result;
		a[5] = angry_result;
		a[6] = neutral_result;

		int temp = a[0];
		int num_temp = 0;
		for(int i = 0; i<6; i++)
		{
			if(a[i+1]>=temp)
			{
				temp = a[i+1];
				num_temp = i+1;
			}
		}

		if(num_temp == 0)
		{
			myfile<<"Disgust";
			myfile.close();
		}
		else if(num_temp == 1)
		{
			myfile<<"Surprise";
			myfile.close();
		}
		else if(num_temp == 2)
		{
			myfile<<"Fear";
			myfile.close();
		}
		else if(num_temp == 3)
		{
			myfile<<"Sad";
			myfile.close();
		}
		else if(num_temp == 4)
		{
			myfile<<"Happy";
			myfile.close();
		}
		else if(num_temp == 5)
		{
			myfile<<"Angry";
			myfile.close();
		}
		else if(num_temp == 6)
		{
			myfile<<"Neutral";
			myfile.close();
		}
	
		ffmpeg.UnLoad();
	}
	return 0;
}


void face_detection_recognition( Mat& img, double scale)
// This function detects face image in a picture and recognize the emotion
{

	// image preprocessing
	Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );// resize the image so as to accelerate the process
	cvtColor( img, gray, CV_BGR2GRAY );
	resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );//resize to 1/scale
	equalizeHist( smallImg, smallImg );

	int * pResults = NULL;
	Rect select;
	pResults = facedetect_multiview_reinforce((unsigned char*)(smallImg.ptr(0)), smallImg.cols, smallImg.rows, smallImg.step,1.2f, 5, 24);
	
	for(int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults+1))+6*i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];
		select.x=p[0]*scale;
		select.y=p[1]*scale;
		select.width=p[2]*scale;
		select.height=p[3]*scale;
		select&=Rect(0,0,gray.cols,gray.rows);//保证所选矩形框在视频显示区域之内
		if (select.width==0 || select.height==0)
		{
			continue;
		}

        // circle the face image using a rectangular box
		cvSetImageROI(image1,select); 
		IplImage *image2 = cvCreateImage(cvSize(select.width,select.height),image1->depth,1); 
		cvCvtColor(image1,image2,CV_BGR2GRAY);
		cvResetImageROI(image1);
		
		// Image normalization
		IplImage *image_r = NULL;
		image_r = cvCreateImage(cvSize(128,128), image1->depth, 1);
		cvResize(image2, image_r, CV_INTER_LINEAR);
		
		// Emotion Recognition
		emotion_recognition_HOG(image_r,select);

		cvRectangle(image1,cvPoint(select.x,select.y),cvPoint(select.x+select.width,select.y+select.height),CV_RGB(0,0,255),2);
		cvReleaseImage(&image2);
		cvReleaseImage(&image_r);
	}
}

void emotion_recognition_LBP(IplImage *image2, Rect select)
// This fucntion implements emotion recognition via LBF-HF && SVM
{

	double final_vector[feature_num]={0.0}; 
	int result = 1;
	CvMat *testMat = cvCreateMat(1, feature_num, CV_32FC1);
	float temp = 1.0f;

	// Extract LBF-HF feature
	get_vector_AU(image2,final_vector);
	for (int i = 0; i<feature_num; i++)
	{
		cvSetReal2D(testMat, 0, i, final_vector[i]); //把提取的识别向量赋值进去
	}
	
	// SVM predicts the category of face emotion
	float flag_happy = 0;
	flag_happy = rtree_happy.predict(testMat);
	cout << "happy:" << flag_happy << endl;
	if (flag_happy<temp)// && flag_happy<0)
	{
		temp = flag_happy;
		result = 2;
	}

	float flag_surprise = 0;
	flag_surprise = rtree_surprise.predict(testMat);
	cout << "surprise:" << flag_surprise << endl;
	if (flag_surprise<temp)// && flag_surprise<0)
	{
		temp = flag_surprise;
		result = 3;
	}

	float flag_angry = 0;
	flag_angry = rtree_angry.predict(testMat);
	cout << "angry:" << flag_angry << endl;
	if (flag_angry<temp)// && flag_angry<0)
	{
		temp = flag_angry;
		result = 4;
	}

	float flag_disgust = 0;
	flag_disgust = rtree_disgust.predict(testMat);
	cout << "disgust:" << flag_disgust << endl;
	if (flag_disgust<temp)// && flag_disgust<0)
	{
		temp = flag_disgust;
		result = 5;
	}

	float flag_fear = 0;
	flag_fear = rtree_fear.predict(testMat);
	cout << "fear:" << flag_fear << endl;
	if (flag_fear<temp)// && flag_fear<0)
	{
		temp=flag_fear;
		result=6;
	}

	float flag_sad = 0;
	flag_sad = rtree_sad.predict(testMat);
	cout << "sad:" << flag_sad << endl;
	if (flag_sad<temp)// && flag_sad<0)
	{
		temp=flag_sad;
		result=7;
	}

	float flag_neutral = 0;
	flag_neutral = rtree_neutral.predict(testMat);
	cout << "neutral:" << flag_neutral << endl;
	if (flag_neutral<temp)// && flag_sad<0)
	{
		temp = flag_neutral;
		result = 8;
	}

	// show the final prediction result on the image
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX,1,0.5,0,2,8);
	if (result==2)
	{
		cvPutText(image1,"happy",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		happy_result++;
	}
	else if (result==3)
	{
		cvPutText(image1,"surprise",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		surprise_result++;
	}
	else if (result==4)
	{
		cvPutText(image1,"angry",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		angry_result++;
	}
	else if (result==5)
	{
		cvPutText(image1,"disgust",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		disgust_result++;
	}
	else if (result==6)
	{
		cvPutText(image1,"fear",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		fear_result++;
	}
	else if (result==7)
	{
		cvPutText(image1,"sad",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		sad_result++;
	}
	else if (result==8)
	{
		cvPutText(image1,"neutral",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		neutral_result++;
	}
}

void emotion_recognition_HOG(IplImage *image2, Rect select)
// This fucntion implements emotion recognition via HOG && SVM
{

	Mat testImg = cvarrToMat(image2);
	vector<float> descriptor;

	// computing HOG descriptor, the window size is (8,8)
	HOGDescriptor hog(Size(128,128),Size(16,16),Size(8,8),Size(8,8),9);
	hog.compute(testImg,descriptor,Size(8,8));
	int DescriptorDim = descriptor.size();

	CvMat *testMat = cvCreateMat(1, DescriptorDim, CV_32FC1);

	for (int i = 0; i < DescriptorDim; i++)
	{
		cvSetReal2D(testMat, 0, i, descriptor[i]); //把提取的识别向量赋值进去
	}

	int result = 1;
	float temp = 1.0f;

	// SVM predicts the category of face emotion
	//cout << endl;
	float flag_happy = 0;
	flag_happy = svm_happy.predict(testMat,TRUE);
	//flag_happy = rtree_happy.predict_prob(testMat);
	//cout << "happy:" << flag_happy << endl;
	if (flag_happy<temp)// && flag_happy<0)
	{
		temp = flag_happy;
		result = 2;
	}

	float flag_surprise = 0;
	flag_surprise = svm_surprise.predict(testMat,TRUE);
	//flag_surprise = rtree_surprise.predict_prob(testMat);
	//cout << "surprise:" << flag_surprise << endl;
	if (flag_surprise<temp)// && flag_surprise<0)
	{
		temp = flag_surprise;
		result = 3;
	}

	float flag_angry = 0;
	flag_angry = svm_angry.predict(testMat,TRUE);
	//flag_angry = rtree_angry.predict_prob(testMat);
	//cout << "angry:" << flag_angry << endl;
	if (flag_angry<temp)// && flag_angry<0)
	{
		temp = flag_angry;
		result = 4;
	}

	float flag_disgust = 0;
	flag_disgust = svm_disgust.predict(testMat,TRUE);
	//flag_disgust = rtree_disgust.predict_prob(testMat);
	//cout << "disgust:" << flag_disgust << endl;
	if (flag_disgust<temp)// && flag_disgust<0)
	{
		temp = flag_disgust;
		result = 5;
	}

	float flag_fear = 0;
	flag_fear = svm_fear.predict(testMat,TRUE);
	//flag_fear = rtree_fear.predict_prob(testMat);
	//cout << "fear:" << flag_fear << endl;
	if (flag_fear<temp)// && flag_fear<0)
	{
		temp = flag_fear;
		result = 6;
	}

	float flag_sad = 0;
	flag_sad = svm_sad.predict(testMat,TRUE);
	//flag_sad = rtree_sad.predict_prob(testMat);
	//cout << "sad:" << flag_sad << endl;
	if (flag_sad<temp)// && flag_sad<0)
	{
		temp = flag_sad;
		result = 7;
	}

	float flag_neutral = 0;
	flag_neutral = svm_neutral.predict(testMat,TRUE);
	//flag_neutral = rtree_neutral.predict_prob(testMat);
	//cout << "neutral:" << flag_neutral << endl;
	if (flag_neutral<temp)// && flag_sad<0)
	{
		temp = flag_neutral;
		result = 8;
	}


	// show the final prediction result on the image
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX,1,0.5,0,2,8);
	if (result==2)
	{
		//cvPutText(image1,"happy",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		happy_result++;
	}
	else if (result==3)
	{
		//cvPutText(image1,"surprise",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		surprise_result++;
	}
	else if (result==4)
	{
		//cvPutText(image1,"angry",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		angry_result++;
	}
	else if (result==5)
	{
		//cvPutText(image1,"disgust",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		disgust_result++;
	}
	else if (result==6)
	{
		//cvPutText(image1,"fear",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		fear_result++;
	}
	else if (result==7)
	{
		//cvPutText(image1,"sad",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		sad_result++;

	}
	else if (result==8)
	{
		//cvPutText(image1,"neutral",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		neutral_result++;
	}
}

void emotion_recognition_LBP_HOG(IplImage *image2, Rect select)
// This fucntion implements emotion recognition via SVM-HF+HOG && SVM
{
	double final_vector[feature_num]={0.0}; 

	// Extract LBF-HF feature
	get_vector_AU(image2,final_vector);

	Mat testImg = cvarrToMat(image2);
	vector<float> descriptor;

	// computing HOG descriptor, the window size is (8,8)
	HOGDescriptor hog(Size(128,128),Size(16,16),Size(8,8),Size(8,8),9);
	hog.compute(testImg,descriptor,Size(8,8));
	int DescriptorDim = descriptor.size();

	CvMat *testMat = cvCreateMat(1, feature_num+DescriptorDim, CV_32FC1);
		
	for (int i = 0; i<feature_num; i++)
	{
		cvSetReal2D(testMat, 0, i, final_vector[i]); 
	}

	for (int i = feature_num; i < feature_num + DescriptorDim; i++)
	{
		cvSetReal2D(testMat, 0, i, descriptor[i-feature_num]); 
	}

	int result = 1;
	float temp = 1.0f;

	// SVM predicts the category of face emotion
	cout << endl;
	float flag_happy = 0;
	flag_happy = svm_happy.predict(testMat,TRUE);
	//flag_happy = rtree_happy.predict_prob(testMat);
	cout << "happy:" << flag_happy << endl;
	if (flag_happy<temp)// && flag_happy<0)
	{
		temp = flag_happy;
		result = 2;
	}

	float flag_surprise = 0;
	flag_surprise = svm_surprise.predict(testMat,TRUE);
	//flag_surprise = rtree_surprise.predict_prob(testMat);
	cout << "surprise:" << flag_surprise << endl;
	if (flag_surprise<temp)// && flag_surprise<0)
	{
		temp = flag_surprise;
		result = 3;
	}

	float flag_angry = 0;
	flag_angry = svm_angry.predict(testMat,TRUE);
	//flag_angry = rtree_angry.predict_prob(testMat);
	cout << "angry:" << flag_angry << endl;
	if (flag_angry<temp)// && flag_angry<0)
	{
		temp = flag_angry;
		result = 4;
	}

	float flag_disgust = 0;
	flag_disgust = svm_disgust.predict(testMat,TRUE)*10;
	//flag_disgust = rtree_disgust.predict_prob(testMat);
	cout << "disgust:" << flag_disgust << endl;
	if (flag_disgust<temp)// && flag_disgust<0)
	{
		temp = flag_disgust;
		result = 5;
	}

	float flag_fear = 0;
	flag_fear = svm_fear.predict(testMat,TRUE);
	//flag_fear = rtree_fear.predict_prob(testMat);
	cout << "fear:" << flag_fear << endl;
	if (flag_fear<temp)// && flag_fear<0)
	{
		temp = flag_fear;
		result = 6;
	}

	float flag_sad = 0;
	flag_sad = svm_sad.predict(testMat,TRUE);
	//flag_sad = rtree_sad.predict_prob(testMat);
	cout << "sad:" << flag_sad << endl;
	if (flag_sad<temp)// && flag_sad<0)
	{
		temp = flag_sad;
		result = 7;
	}

	float flag_neutral=0;
	flag_neutral = svm_neutral.predict(testMat,TRUE);
	//flag_neutral = rtree_neutral.predict_prob(testMat);
	cout << "neutral:" << flag_neutral << endl;
	if (flag_neutral<temp)// && flag_sad<0)
	{
		temp = flag_neutral;
		result = 8;
	}
	cout << endl;

	// show the final prediction result on the image
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX,1,0.5,0,2,8);
	if (result==2)
	{
		cvPutText(image1,"happy",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		happy_result++;
	}
	else if (result==3)
	{
		cvPutText(image1,"surprise",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		surprise_result++;
	}
	else if (result==4)
	{
		cvPutText(image1,"angry",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		angry_result++;
	}
	else if (result==5)
	{
		cvPutText(image1,"disgust",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		disgust_result++;
	}
	else if (result==6)
	{
		cvPutText(image1,"fear",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		fear_result++;
	}
	else if (result==7)
	{
		cvPutText(image1,"sad",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		sad_result++;
	}
	else if (result==8)
	{
		cvPutText(image1,"neutral",cvPoint(select.x,select.y),&font,CV_RGB(255,0,0));
		neutral_result++;
	}
}