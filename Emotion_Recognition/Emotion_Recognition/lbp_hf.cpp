
//This file defines the functions for extracting LBF-HF features of face images
//Author: Liang Li
//LBP-HF is proposed in 
//               T. Ahonen, J. Matas, C. He, M. Pietikäinen. Rotation Invariant
//               Image Description with Local Binary Pattern Histogram
//               Fourier Features. Image Analysis, 2015, 5575(4):61-70.
//

#include "cv.h"
#include "ml.h"
#include "highgui.h"
#include "math.h"
#include "iostream"
#include <stdio.h>
#include <math.h>
#include <cxcore.h>

using namespace std;
using namespace cv;

#define  feature_num 1110 //number of features of LBF-HF descriptor

int hist[58] = {0};
int total_num = 0;
int rotate_map[7][8] = {{1,2,4,8,16,32,64,128},
						{3,6,12,24,48,96,192,129},
						{131,7,14,28,56,112,224,193},
						{135,15,30,60,120,240,225,195},
						{199,143,31,62,124,248,241,227},
						{207,159,63,126,252,249,243,231},
						{239,223,191,127,254,253,251,247}};

int AU_feature[30]={9,10,11,12,13,14,17,
					18,21,22,33,34,37,38,
					41,42,43,44,45,46,49,
					50,51,52,53,54,58,59,
					60,61};  // index of 8*8 blocks in the image


#define N	8				// length of the sequence 
#define PI	3.1415926535	
typedef double ElementType;

typedef struct				//complex structure for Fourier operator 
{
	ElementType real,imag;
}complex_lbp;
complex_lbp dataResult[N];		 // the value of the sequence in frequency domain
ElementType dataSource[N];	     // primitive input data sequence
ElementType dataFinualResult[N]; // final value of the sequence in frequency domain

void FFT_Calculate_OneNode(int k)
// calculate the DFT value of a point in the frequency domain
{
	int n = 0;
	complex_lbp ResultThisNode;
	complex_lbp part[N];
	ResultThisNode.real = 0;
	ResultThisNode.imag = 0;
	for(n=0; n<N; n++)
	{
		// real part and imaginary part(Eular equation) 
		part[n].real = cos(2*PI/N*k*n)*dataSource[n];
		part[n].imag = -sin(2*PI/N*k*n)*dataSource[n];

		ResultThisNode.real += part[n].real;
		ResultThisNode.imag += part[n].imag;
	}
	dataResult[k].real = ResultThisNode.real;
	dataResult[k].imag = ResultThisNode.imag;
}

void FFT_Calculate()
//calculate DFT for all the input sequence
{
	int i = 0;
	for(i=0; i<N; i++)
	{
		FFT_Calculate_OneNode(i);
		dataFinualResult[i] = sqrt(dataResult[i].real * dataResult[i].real + dataResult[i].imag * dataResult[i].imag);
	}
}

void re_initial()
{
	for (int j=0;j<58;j++)
	{
		hist[j]=0;
	}
	total_num=0;
	for (int i=0;i<N;i++)
	{
		dataResult[i].imag=0.0;	
		dataResult[i].real=0.0;	// final value of the sequence in frequency domain
		dataSource[i]=0.0;	    // primitive input data sequence
		dataFinualResult[i]=0.0;
	}
}


void LBP_HF(IplImage* image, int num)
// extract LBP-HF feature
{
	int width  = image->width;
	int height = image->height;

	int col_num=num%8;
	int row_num=num/8;

	for (int i = 1+height/8*row_num; i <height/8*(row_num+1)-1; i++)
		for (int j = 1+width/8*col_num; j <width/8*(col_num+1)-1; j++)
		{
			total_num++;
			int center = (unsigned char)image->imageData[i*width + j];
			unsigned char result = 0;
			int temp = 0;

			temp = (unsigned char)image->imageData[(i - 1)*width + j - 1];
			//if (abs(temp - center)>gap)
			if (temp>center)
			{
				result += 1;
			}
			temp = (unsigned char)image->imageData[(i - 1)*width + j];
			if (temp>center)
			{
				result += 2;
			}
			temp = (unsigned char)image->imageData[(i - 1)*width + j + 1];
			if (temp>center)
			{
				result += 4;
			}
			temp= (unsigned char)image->imageData[i*width + j + 1];
			if (temp>center)
			{
				result += 8;
			}
			temp = (unsigned char)image->imageData[(i+1)*width + j + 1];
			if (temp>center)
			{
				result += 16;
			}
			temp = (unsigned char)image->imageData[(i + 1)*width + j ];
			if (temp>center)
			{
				result += 32;
			}
			temp = (unsigned char)image->imageData[(i + 1)*width + j-1];
			if (temp>center)
			{
				result += 64;
			}
			temp = (unsigned char)image->imageData[i*width + j - 1];
			if (temp>center)
			{
				result += 128;
			}	

			if (result == 0)
			{
				hist[0]++;
			}
			else if (result == 255)
			{
				hist[57]++;
			}
			else
			{
				unsigned char temp = result;
				unsigned char img_data = result;
				unsigned char temp_1 = result;
				if (result>=128)
				{
					result = result << 1;
					result +=1;
				}
				else
				{
					result = result << 1;
				}

				result = result^temp;

				int count = 0;
				int bit_count = 0;
				while (result > 0)
				{
					result = result&(result - 1);
					count++;
				}
				while (temp > 0)
				{
					temp = temp&(temp - 1);
					bit_count++;
				}
				if (count == 2)
				{
					for (int i=0;i<8;i++)
					{
						if (rotate_map[bit_count-1][i]==temp_1)
						{
							int map_index=(bit_count-1)*8+1+i;
							hist[map_index]++;
							break;
						}
					}
				}
			}
		}

}

void get_vector_AU(IplImage* image,double final_vector[])
// extract LBP-HF feature && normalization
{

	for (int block_num_final=0;block_num_final<30;block_num_final++)
	{
		re_initial();
		LBP_HF(image,AU_feature[block_num_final]);
		for (int i=0;i<7;i++)
		{
			for (int j=1+i*8;j<9+i*8;j++)
			{
				dataSource[j-1-i*8]=hist[j]*1.0/total_num; //calculate the 8 uniform LBP in a row
			}
			FFT_Calculate();
			for(int k=0; k<5; k++)
			{
				final_vector[block_num_final*37+i*5+k]=dataFinualResult[k];
			}
		}
		// normalization
		final_vector[block_num_final*37+35]=hist[0]*1.0/total_num;
		final_vector[block_num_final*37+36]=hist[57]*1.0/total_num;
	}
}