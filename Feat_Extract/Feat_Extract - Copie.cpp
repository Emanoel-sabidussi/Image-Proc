#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include <opencv2/core/cxcore.h>
#include <math.h>

#include "Feat_Extract.hpp"

using namespace std;

void Feat_Extract::initEverything() {

	for (int i = 0; i < 1; i++){

		histog(dest_, &hist_graph);

		std_mean(i, &Gabor_mean, &Gabor_std);
		
		energy(i, &Gabor_energy); // FIX!!! this is the same as Mean (with the implemented code);
		
		entropy(dest_, &Gabor_entropy, 1);

		GLCM(dest_, &glcm);

		entropy(glcm, &glcm_entropy, 0);

		ASM(glcm, &ASM_);
		
		Correl(glcm, &corr_);

		IDM(glcm, &IDM_);

	}
	
	feat_form(&feat_mat);

	cout << "feat mat: "<< endl <<  feat_mat << endl;

}

void Feat_Extract::std_mean(int i, double *Gabor_mean, double *Gabor_std){
	
	Scalar mean_sc, std_sc;
	
	meanStdDev(dest_, mean_sc, std_sc);

	//Gabor_mean->push_back(mean_sc);

	*Gabor_mean = mean_sc[0];
	*Gabor_std = std_sc[0];

	//Gabor_std->push_back(std_sc);

}

void Feat_Extract::energy(int i, double *Gabor_energy){


	Mat Itemp;
	Scalar energy;

	Size size = dest_.size();
	Scalar total_size = size.height * size.width;

	pow(dest_, 2, Itemp);

	energy = sum(sum(Itemp)) / total_size;

//	Gabor_energy->push_back(energy);

	*Gabor_energy = energy[0];

}

void Feat_Extract::entropy(Mat dest, double *Gabor_entropy, int flag){

	Mat Itemp;
	Scalar entropy;

	Size size = dest.size();
	Scalar total_size = size.height * size.width;

	dest.convertTo(dest, CV_32FC3);

	Itemp = (dest + 1)* (1 / log(10));
	
	Itemp = dest.mul(Itemp);

	//multiply(dest, Itemp, Itemp);
	if (flag > 0)
		entropy = sum(sum(Itemp)) / total_size;
	else
		entropy = sum(sum(Itemp));

	//Gabor_entropy->push_back(entropy);

	*Gabor_entropy = entropy[0];



}

void Feat_Extract::histog(Mat Input, vector<Mat> *hist_graph){

	Mat hist_im;
	hist_im = Input;

	hist_im.convertTo(hist_im, CV_8UC1, 255);

	int histSize = 255;

	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	calcHist(&hist_im, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	hist_graph->push_back(histImage);
}

void Feat_Extract::GLCM(Mat Input, Mat *glcm){

	Mat temp_glcm = Mat::zeros(1000, 1000, CV_32FC1);
	Mat temp_glcm0 = Mat::zeros(1000, 1000, CV_32FC1);
	Mat temp_glcm45 = Mat::zeros(1000, 1000, CV_32FC1);
	Mat temp_glcm90 = Mat::zeros(1000, 1000, CV_32FC1);
	Mat temp_glcm135 = Mat::zeros(1000, 1000, CV_32FC1);

	float intI;
	float intJ;
	float intJBack;
	double max, min;

	int dist;
	dist = 4;

	minMaxLoc(Input, &min, &max);

	//0 degrees
	for (int i = 0; i < Input.rows; i++){
		for (int j = dist; j < Input.cols - dist; j++){

			intI = Input.at<float>(i, j);
			intJ = Input.at<float>(i, j + dist);
			intJBack = Input.at<float>(i, j - 4);

			intI = round(255 * intI / (float)max);
			intJ = round(255 * intJ / (float)max);
			
			intJBack = round(255 * intJBack / (float)max);

			if (intI < 0){
				intI = 0;
			}
			if (intJ < 0){
				intJ = 0;
			}
			if (intJBack < 0){
				intJBack = 0;
			}

			temp_glcm0.at<float>(intI, intJ) = temp_glcm0.at<float>(intI, intJ) + 1;
			temp_glcm0.at<float>(intI, intJBack) = temp_glcm0.at<float>(intI, intJBack) + 1;
		}
	}
	//45 degrees
	for (int i = dist; i < Input.rows - dist; i++){
		for (int j = dist; j < Input.cols - dist; j++){

			intI = Input.at<float>(i, j);
			intJ = Input.at<float>(i + dist, j - dist);
			intJBack = Input.at<float>(i - dist, j + dist);

			intI = round(255 * intI / (float)max);
			intJ = round(255 * intJ / (float)max);
			intJBack = round(255 * intJBack / (float)max);

			if (intI < 0){
				intI = 0;
			}
			if (intJ < 0){
				intJ = 0;
			}
			if (intJBack < 0){
				intJBack = 0;
			}

			temp_glcm45.at<float>(intI, intJ) = temp_glcm45.at<float>(intI, intJ) + 1;
			temp_glcm45.at<float>(intI, intJBack) = temp_glcm45.at<float>(intI, intJBack) + 1;
		}
	}
	//90 Degrees
	for (int i = dist; i < Input.rows - dist; i++){
		for (int j = 0; j < Input.cols; j++){

			intI = Input.at<float>(i, j);
			intJ = Input.at<float>(i + dist, j);
			intJBack = Input.at<float>(i - dist, j);

			intI = round(255 * intI / (float)max);
			intJ = round(255 * intJ / (float)max);
			intJBack = round(255 * intJBack / (float)max);


			if (intI < 0){
				intI = 0;
			}
			if (intJ < 0){
				intJ = 0;
			}
			if (intJBack < 0){
				intJBack = 0;
			}

			temp_glcm90.at<float>(intI, intJ) = temp_glcm90.at<float>(intI, intJ) + 1;
			temp_glcm90.at<float>(intI, intJBack) = temp_glcm90.at<float>(intI, intJBack) + 1;
		}
	}
	//135 Degrees
	for (int i = dist; i < Input.rows - dist; i++){
		for (int j = dist; j < Input.cols - dist; j++){

			intI = Input.at<float>(i, j);
			intJ = Input.at<float>(i + dist, j + dist);
			intJBack = Input.at<float>(i - dist, j - dist);

			intI = round(255 * intI / (float)max);
			intJ = round(255 * intJ / (float)max);
			intJBack = round(255 * intJBack / (float)max);

			if (intI < 0){
				intI = 0;
			}
			if (intJ < 0){
				intJ = 0;
			}
			if (intJBack < 0){
				intJBack = 0;
			}

			temp_glcm135.at<float>(intI, intJ) = temp_glcm135.at<float>(intI, intJ) + 1;
			temp_glcm135.at<float>(intI, intJBack) = temp_glcm135.at<float>(intI, intJBack) + 1;
		}
	}

	float Sum = float(Input.cols * (Input.rows - 1));
	temp_glcm = (temp_glcm0 + temp_glcm45 + temp_glcm90 + temp_glcm135)/4;
	//pow(temp_glcm, (1 / 4), temp_glcm);
	temp_glcm = temp_glcm / (2);
	

	//glcm->push_back(temp_glcm);

	*glcm = temp_glcm;

	//cout << "glcm: " << *glcm << endl;

	/*temp_glcm.release();
	temp_glcm0.release();
	temp_glcm45.release();
	temp_glcm90.release();
	temp_glcm135.release();*/

}

void Feat_Extract::ASM(Mat GLCM, double *ASM_){

	Mat glcm_temp;
	Scalar asm_temp;

	//pow(GLCM, 2, glcm_temp);
	glcm_temp = GLCM.mul(GLCM);

	asm_temp = sum(sum(glcm_temp));
	//ASM_->push_back(asm_temp);
	*ASM_ = asm_temp[0];

}

void Feat_Extract::Correl(Mat GLCM, double *corr_){

	Scalar corr_temp;
	float Uj = 0;
	float Ui = 0;
	float Sj = 0;
	float Si = 0;


	for (int i = 1; i < GLCM.rows; i++){
		for (int j = 1; j < GLCM.cols; j++){

			Uj = Uj + j*GLCM.at<float>(i, j);
			Ui = Ui + i*GLCM.at<float>(i, j);
		}
	}

	for (int i = 1; i < GLCM.rows; i++){
		for (int j = 1; j < GLCM.cols; j++){

			Sj = Sj + pow((j - Uj), 2)*GLCM.at<float>(i, j);
			Si = Si + pow((i - Ui), 2)*GLCM.at<float>(i, j);
		}
	}

	for (int i = 1; i < GLCM.rows; i++){
		for (int j = 1; j < GLCM.cols; j++){

			corr_temp[0] = corr_temp[0] + ((i - Ui)*(j - Uj)*GLCM.at<float>(i, j)) / ((Si*Sj) + 1);
		}
	}

	//corr_->push_back(corr_temp[0]);

	*corr_ = corr_temp[0];


}

void Feat_Extract::IDM(Mat GLCM, double *IDM_){

	Scalar idm_temp = 0;

	for (int i = 1; i < GLCM.rows; i++){
		for (int j = 1; j < GLCM.cols; j++){

			idm_temp[0] = idm_temp[0] + (GLCM.at<float>(i, j) / (1 + pow((i - j), 2)));
		}
	}
	//IDM_->push_back(idm_temp[0]);

	*IDM_ = idm_temp[0];
}

void Feat_Extract::feat_form(Mat *feat_mat){

	for (int j = 0; j < 1; j++){
		
		feat_vec->push_back(Gabor_mean);//*feat_vec = temp; feat_vec++;

		feat_vec->push_back(Gabor_std);//*feat_vec = temp; feat_vec++;
		
		feat_vec->push_back(Gabor_energy);//*feat_vec = temp; feat_vec++;		
		
		feat_vec->push_back(Gabor_entropy);//*feat_vec = temp; feat_vec++;
		
		feat_vec->push_back(ASM_);//*feat_vec = temp; feat_vec++;
		
		feat_vec->push_back(glcm_entropy);//*feat_vec = temp; feat_vec++;
		
		feat_vec->push_back(corr_);//*feat_vec = temp; feat_vec++;
		
		feat_vec->push_back(IDM_);//*feat_vec = temp; feat_vec++;
		

	}

	memcpy(feat_mat->data, FEAT.data(), FEAT.size()*sizeof(CV_32FC1));
}