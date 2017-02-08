#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include <opencv2/core/cxcore.h>

#include "GaborFilter.hpp"

using namespace std;

void GaborFilter::initEverything() {

	vector<Mat> Kernel;
	vector<Mat> Kernel_ph;
	vector<Mat> dest(37);
	vector<Mat> dest_ph(37);

	int co = 0;
	int ro = 0;

	get_kernel(&Kernel, &Kernel_ph, input_);
	input_.convertTo(input_, CV_32F, 0.5/ 255, 0);

	Mat test = Mat::zeros(input_.rows, input_.cols, CV_32F);
	Mat test1;

	for (int i = 0; i < 36; i++){
		filter2D(input_, dest[i], CV_32F, Kernel[i]);
		filter2D(input_, dest_ph[i], CV_32F, Kernel_ph[i]);

		Gabor.push_back(dest[i]);
		Gabor_ph.push_back(dest_ph[i]);

		//imshow("gabor", dest[i]);
		//waitKey(0);

	}

	for (int i = 0; i < 18; i++){
		multiply(dest[i*2], dest_ph[i*2], test1);
		//test = test + (dest[i] + dest_ph[i]) /4;
		test = (test + test1) / 2;
	}
	

	Final_gabor = test;

}

void GaborFilter::get_kernel(vector<Mat> *Kernel, vector<Mat> *Kernel_ph, Mat Clahe_IM){

	double gamma = 1;
	double psi = 0;
	double psi_ph = 90;
	double lambda = 0;
	double sigma = 0;
	double theta = 0;
	int ks = 30;

	for (int i = 0; i < 12; i++){

		lambda = 5.24;
		sigma = 0.56*lambda;
		ks = 5;
		Kernel->push_back(GaborFilt(/*theta*/ theta, /*sigma*/ sigma, /*lambda*/ lambda, /*gamma*/ gamma, /*psi*/ psi, i, ks));
		Kernel_ph->push_back(GaborFilt(/*theta*/ theta, /*sigma*/ sigma, /*lambda*/ lambda, /*gamma*/ gamma, /*psi*/ psi_ph, i, ks));

		lambda = 4.12;
		sigma = 0.56*lambda;
		ks = 7;
		Kernel->push_back(GaborFilt(/*theta*/ theta, /*sigma*/ sigma, /*lambda*/ lambda, /*gamma*/ gamma, /*psi*/ psi, i, ks));
		Kernel_ph->push_back(GaborFilt(/*theta*/ theta, /*sigma*/ sigma, /*lambda*/ lambda, /*gamma*/ gamma, /*psi*/ psi_ph, i, ks));

		lambda = 2.56;
		sigma = 0.56*lambda;
		ks = 9;
		Kernel->push_back(GaborFilt(/*theta*/ theta, /*sigma*/ sigma, /*lambda*/ lambda, /*gamma*/ gamma, /*psi*/ psi, i, ks));
		Kernel_ph->push_back(GaborFilt(/*theta*/ theta, /*sigma*/ sigma, /*lambda*/ lambda, /*gamma*/ gamma, /*psi*/ psi_ph, i, ks));

		theta = theta + 15;
	}
}

Mat GaborFilter::GaborFilt(double pos_th, double Sigma, double Lambda, double Gamma, double psi, int i, int ks){

	Mat Kernel;

	psi = psi * CV_PI / 180;
	double theta = pos_th*CV_PI / 180;

	Size KernalSize(ks, ks);
	Kernel = getGaborKernel(KernalSize, Sigma, theta, Lambda, Gamma, psi);

	return Kernel;
}

