#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/plot.hpp"
#include "opencv2/photo.hpp"


#include <iostream>
#include <fstream>
#include <iterator>
#include <stdlib.h>
#include <stdio.h>

#include "Wavelet.hpp"

using namespace std;
using namespace cv;

// Filter type
#define NONE 0  // no filter
#define HARD 1  // hard shrinkage
#define SOFT 2  // soft shrinkage
#define GARROT 3  // garrot filter
//--------------------------------
// signum
//--------------------------------


void Wavelet::initEverything() {


	cout << "passed" << endl;

	cvHaarWavelet();
	imshow("Coeff111", dst);
	waitKey(0);

	//double M = 0, m = 0;
	//----------------------------------------------------
	// Normalization to 0-1 range (for visualization)
	//----------------------------------------------------
	//minMaxLoc(dst, &m, &M);
	//if ((M - m)>0) { dst = dst*(1.0 / (M - m)) - m / (M - m); }
	//imshow("Coeff", dst);

}

float Wavelet::sgn(float x)
{
	float res = 0;
	if (x == 0)
	{
		res = 0;
	}
	if (x>0)
	{
		res = 1;
	}
	if (x<0)
	{
		res = -1;
	}
	return res;
}

//--------------------------------
// Soft shrinkage
//--------------------------------
float Wavelet::soft_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = sgn(d)*(fabs(d) - T);
	}
	else
	{
		res = 0;
	}

	return res;
}
//--------------------------------
// Hard shrinkage
//--------------------------------
float Wavelet::hard_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = d;
	}
	else
	{
		res = 0;
	}

	return res;
}
//--------------------------------
// Garrot shrinkage
//--------------------------------
float Wavelet::Garrot_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = d - ((T*T) / d);
	}
	else
	{
		res = 0;
	}

	return res;
}

//--------------------------------
// Wavelet transform
//--------------------------------
void Wavelet::cvHaarWavelet()
{
	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;

	Mat dst_t = Mat(src.rows, src.cols, CV_32FC1);

	dst_t.copyTo(dst);

	for (int k = 0; k<NIter; k++)
	{
		for (int y = 0; y<(height >> (k + 1)); y++)
		{
			for (int x = 0; x<(width >> (k + 1)); x++)
			{
				c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			
				dst.at<float>(y, x) = c;

				dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x + (width >> (k + 1))) = dh*5;

				dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x) = dv *5;

				dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd *5;
			}
		}

		dst.copyTo(src);
	}
}







void Wavelet::Wavelet_decomp(){

	double a, b, c, d;
	Mat imd = Mat::zeros((src.rows / 2), (src.cols / 2), CV_32F);
	

	//im1 = Mat::zeros(src.rows, src.cols, CV_32F);
	Mat im1 = Mat::zeros(src.rows / 2, src.cols, CV_32F);
	Mat im2 = Mat::zeros(src.rows / 2, src.cols, CV_32F);
	Mat im3 = Mat::zeros(src.rows / 2, src.cols / 2, CV_32F);
	Mat im4 = Mat::zeros(src.rows / 2, src.cols / 2, CV_32F);
	Mat im5 = Mat::zeros(src.rows / 2, src.cols / 2, CV_32F);
	Mat im6 = Mat::zeros(src.rows / 2, src.cols / 2, CV_32F);

	src.copyTo(im1);

	for (int rcnt = 0; rcnt < src.rows; rcnt += 2){
		for (int ccnt = 0; ccnt < src.cols; ccnt++){
			a = src.at<float>(rcnt, ccnt);
			b = src.at<float>(rcnt + 1, ccnt);
			c = (a + b)*0.707;
			d = (a - b)*0.707;
			int _rcnt = rcnt / 2;
			im1.at<float>(_rcnt, ccnt) = c;
			im2.at<float>(_rcnt, ccnt) = d;
		}
	}

	for (int rcnt = 0; rcnt < src.rows / 2; rcnt++){
		for (int ccnt = 0; ccnt < src.cols; ccnt += 2){
			a = im1.at<float>(rcnt, ccnt);
			b = im1.at<float>(rcnt, ccnt + 1);
			c = (a + b)*0.707;
			d = (a - b)*0.707;
			int _ccnt = ccnt / 2;
			im3.at<float>(rcnt, _ccnt) = c;
			im4.at<float>(rcnt, _ccnt) = d;
		}
	}

	for (int rcnt = 0; rcnt < src.rows / 2; rcnt++){
		for (int ccnt = 0; ccnt < src.cols; ccnt += 2){
			a = im2.at<float>(rcnt, ccnt);
			b = im2.at<float>(rcnt, ccnt + 1);
			c = (a + b)*0.707;
			d = (a - b)*0.707;
			int _ccnt = ccnt / 2;
			im5.at<float>(rcnt, _ccnt) = c;
			im6.at<float>(rcnt, _ccnt) = d;
		}
	}


	cout << "im3: " << im3.size() << endl;
	cout << "im4: " << im4.size() << endl;
	cout << "im5: " << im5.size() << endl;
	cout << "im6: " << im6.size() << endl;
	cout << "imd: " << imd.size() << endl;


	Mat imd_temp = im4 + im5 + im6;

	cout << "imd_temp: " << imd_temp.size() << endl;

	


	//im3.copyTo(imd(Rect(0, 0, src.cols / 2, src.rows / 2)));
	//im4.copyTo(imd(Rect(0, (src.rows / 2) - 1, src.cols / 2, src.rows / 2)));
	//im5.copyTo(imd(Rect((src.cols / 2) - 1, 0, src.cols / 2, src.rows / 2)));
    //im6.copyTo(imd(Rect((src.cols / 2) - 1, (src.rows / 2) - 1, src.cols / 2, src.rows / 2)));


	//im7.copyTo(imd(Rect((src.cols / 4) - 1, 0, src.cols / 4, src.rows / 4)));
	//im8.copyTo(imd(Rect((src.cols / 4) - 1, (src.rows / 4) - 1, src.cols / 4, src.rows / 4)));
	//im9.copyTo(imd(Rect(0, 0, src.cols / 4, src.rows / 4)));
	//im10.copyTo(imd(Rect(0, (src.rows / 4) - 1, src.cols / 4, src.rows / 4)));

		imd_temp.copyTo(imd);

		cout << "add imd: " << &imd << endl;

	//imshow("im711", temp1);
	imshow("WAVELET", imd_temp);
	imshow("imd", imd);
	waitKey(0);

	
}
