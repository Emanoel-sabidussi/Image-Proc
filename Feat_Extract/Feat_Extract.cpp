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

		Laws_filter();


		//histog(dest_, &hist_graph);


		//std_mean(i, &Gabor_mean, &Gabor_std);
		
		//energy(i, &Gabor_energy); // FIX!!! this is the same as Mean (with the implemented code);
		
		//entropy(dest_, &Gabor_entropy, 1);

		//GLCM(dest_, &glcm);

		//entropy(glcm, &glcm_entropy, 0);

		//ASM(glcm, &ASM_);
		
		//Correl(glcm, &corr_);

		//IDM(glcm, &IDM_);


		//Fourier_dft(&dft_entropy, &dft_inertia, &dft_energy);




	}
	
	

	
	

}
/*
void Feat_Extract::std_mean(int i, double *Gabor_mean, double *Gabor_std){
	
	Scalar mean_sc, std_sc;
	
	meanStdDev(hist_graph, mean_sc, std_sc);

	*Gabor_mean = mean_sc[0];
	*Gabor_std = std_sc[0];

}

void Feat_Extract::energy(int i, double *Gabor_energy){


	Mat Itemp;
	Scalar energy;

	Size size = dest_.size();
	Scalar total_size = size.height * size.width;

	pow(dest_, 2, Itemp);

	energy = sum(sum(Itemp)) / total_size;
	
	*Gabor_energy = energy[0];

}

void Feat_Extract::entropy(Mat dest, double *Gabor_entropy, int flag){

	Mat Itemp;
	Scalar entropy;

	Size size = dest.size();
	Scalar total_size = size.height * size.width;

	dest.convertTo(dest, CV_32FC3);
	log((dest + 1), Itemp);
	//Itemp = log(dest +1);
	
	Itemp = dest.mul(Itemp);

	if (flag > 0)
		entropy = sum(sum(Itemp)) / total_size;
	else
		entropy = sum(sum(Itemp));

	*Gabor_entropy = entropy[0];



}

void Feat_Extract::histog(Mat Input, Mat *hist_graph){

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

	*hist_graph = histImage;
	hist_graph->push_back(histImage);
}
*/
/*
void Feat_Extract::GLCM(Mat Input, Mat *glcm){

	Mat temp_glcm = Mat::zeros(500, 500, CV_32FC1);
	Mat temp_glcm0 = Mat::zeros(500, 500, CV_32FC1);
	Mat temp_glcm45 = Mat::zeros(500, 500, CV_32FC1);
	Mat temp_glcm90 = Mat::zeros(500, 500, CV_32FC1);
	Mat temp_glcm135 = Mat::zeros(500, 500, CV_32FC1);

	float intI;
	float intJ;
	float intJBack;
	double max, min;

	int dist;
	dist = 5;

	minMaxLoc(Input, &min, &max);

	//0 degrees
	for (int i = 0; i < Input.rows; i++){
		for (int j = dist; j < Input.cols - dist; j++){

			intI = Input.at<float>(i, j);
			intJ = Input.at<float>(i, j + dist);
			intJBack = Input.at<float>(i, j - dist);

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
			//cout << "int j: " << intJ << endl;
			//cout << "int i: " << intI << endl;
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
			//cout << "int j: " << intJ << endl;
			//cout << "int i: " << intI << endl;
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
			//cout << "int j: " << intJ << endl;
			//cout << "int i: " << intI << endl;
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
			//cout << "int j: " << intJ << endl;
			// cout << "int i: " << intI << endl;
			temp_glcm135.at<float>(intI, intJ) = temp_glcm135.at<float>(intI, intJ) + 1;
			temp_glcm135.at<float>(intI, intJBack) = temp_glcm135.at<float>(intI, intJBack) + 1;
			
		}
	}

	float Sum = float(Input.cols * (Input.rows - 1));
	temp_glcm = (temp_glcm0 + temp_glcm45 + temp_glcm90 + temp_glcm135)/4;

	//imshow("glcm", temp_glcm);
	//temp_glcm = temp_glcm / (2);

	

	minMaxLoc(temp_glcm, &min, &max);


	temp_glcm = temp_glcm / max;
	*glcm = temp_glcm;

}
*/
/*
void Feat_Extract::ASM(Mat GLCM, double *ASM_){

	Mat glcm_temp;
	Scalar asm_temp;

	glcm_temp = GLCM.mul(GLCM);

	asm_temp = sum(sum(glcm_temp));
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

	*corr_ = corr_temp[0];


}

void Feat_Extract::IDM(Mat GLCM, double *IDM_){

	Scalar idm_temp = 0;

	for (int i = 1; i < GLCM.rows; i++){
		for (int j = 1; j < GLCM.cols; j++){

			idm_temp[0] = idm_temp[0] + (GLCM.at<float>(i, j) / (1 + pow((i - j), 2)));
		}
	}

	*IDM_ = idm_temp[0];
}
*/
/*
void Feat_Extract::Fourier_dft(double *dft_entropy, double *dft_inertia, double *dft_energy){


	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(dest_.rows);
	int n = getOptimalDFTSize(dest_.cols); // on the border add zero values
	copyMakeBorder(dest_, padded, 0, m - dest_.rows, 0, n - dest_.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix
	

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	imshow("Input Image", dest_);    // Show the result
	imshow("spectrum magnitude", magI);












	Mat NFP = Mat::zeros(magI.size(), CV_32F);
	Mat temp_nfp = Mat::zeros(magI.size(), CV_32F);

	float temp_val = 0;

	for (int u = 0; u < magI.rows; u++){
		for (int v = 0; v < magI.cols; v++){
			if (u!=v)
				temp_val = temp_val + (magI.at<float>(u, v)*magI.at<float>(u, v));
			
		}
	}




	temp_val = cvSqrt(temp_val);

	NFP = magI / temp_val;

	//divide(double(temp_val), magI, NFP);

	

	temp_val = 0;

	float sum_entropy = 0;

	for (int u = 0; u < NFP.rows; u++){
		for (int v = 0; v < NFP.cols; v++){

			temp_val = log10f(NFP.at<float>(u, v) + 1);
			temp_val = NFP.at<float>(u, v) * temp_val;
			sum_entropy = sum_entropy + temp_val;
			temp_val = 0;
		}
	}

	temp_val = 0;
	float sum_energy = 0;

	for (int u = 0; u < NFP.rows; u++){
		for (int v = 0; v < NFP.cols; v++){

			temp_val = NFP.at<float>(u, v) * NFP.at<float>(u, v);
			sum_energy = sum_energy + temp_val;
			
			//temp_val = 0;
		}
	}


	temp_val = 0;
	float sum_inertia = 0;

	for (int u = 0; u < NFP.rows; u++){
		for (int v = 0; v < NFP.cols; v++){

			temp_val = NFP.at<float>(u, v) * ((u-v)*(u-v));
			sum_inertia = sum_inertia + temp_val;

			//temp_val = 0;
		}
	}


	*dft_entropy = double(sum_entropy);
	*dft_inertia = double(sum_inertia);
	*dft_energy = double(sum_energy);
}*/

void Feat_Extract::Laws_filter(){

	Mat L5 = (Mat_<double>(1, 5) << 1, 4, 6, 4, 1);
	Mat E5 = (Mat_<double>(1, 5) << -1, -2, 0, 2, 1);
	Mat S5 = (Mat_<double>(1, 5) << -1, 0, 2, 0, -1);
	Mat W5 = (Mat_<double>(1, 5) << -1, 2, 0, -2, 1);
	Mat R5 = (Mat_<double>(1, 5) << 1, -4, 6, -4, 1);


	Mat L5T; transpose(L5, L5T);	
	Mat E5T; transpose(E5, E5T);
	Mat S5T; transpose(S5, S5T);
	Mat W5T; transpose(W5, W5T);
	Mat R5T; transpose(R5, R5T);

	Mat dest_Law;

	Mat FML5L5;	
	FML5L5 = L5T * L5;
	filter2D(dest_, dest_Law, CV_32F, FML5L5);
	Get_TEM(dest_Law);
	cout << "FML5L5" << endl;

	Mat FML5E5;	
	FML5E5 = L5T * E5;
	filter2D(dest_, dest_Law, CV_32F, FML5E5);
	Get_TEM(dest_Law);
	cout << "FML5E5" << endl;

	Mat FME5L5;	
	FME5L5 = E5T * L5;
	filter2D(dest_, dest_Law, CV_32F, FME5L5);
	Get_TEM(dest_Law);
	cout << "FME5L5" << endl;

	Mat FML5S5;	
	FML5S5 = L5T * S5;
	filter2D(dest_, dest_Law, CV_32F, FML5S5);
	Get_TEM(dest_Law);
	cout << "FML5S5" << endl;

	Mat FMS5L5;	
	FMS5L5 = S5T * L5;
	filter2D(dest_, dest_Law, CV_32F, FMS5L5);
	Get_TEM(dest_Law);
	cout << "FMS5L5" << endl;

	Mat FML5W5;	
	FML5W5 = L5T * W5;
	filter2D(dest_, dest_Law, CV_32F, FML5W5);
	Get_TEM(dest_Law);
	cout << "FML5W5" << endl;

	Mat FMW5L5;
	FMW5L5 = W5T * L5;
	filter2D(dest_, dest_Law, CV_32F, FMW5L5);
	Get_TEM(dest_Law);
	cout << "FMW5L5" << endl;

	Mat FML5R5;
	FML5R5 = L5T * R5;
	filter2D(dest_, dest_Law, CV_32F, FML5R5);
	Get_TEM(dest_Law);
	cout << "FML5R5" << endl;

	Mat FMR5L5;	
	FMR5L5 = R5T * L5;
	filter2D(dest_, dest_Law, CV_32F, FMR5L5);
	Get_TEM(dest_Law);
	cout << "FMR5L5" << endl;

	Mat FME5E5;
	FME5E5 = E5T * E5;
	filter2D(dest_, dest_Law, CV_32F, FME5E5);
	Get_TEM(dest_Law);
	cout << "FME5E5" << endl;

	Mat FME5S5;
	FME5S5 = E5T * S5;
	filter2D(dest_, dest_Law, CV_32F, FME5S5);
	Get_TEM(dest_Law);
	cout << "FME5S5" << endl;

	Mat FMS5E5;
	FMS5E5 = S5T * E5;
	filter2D(dest_, dest_Law, CV_32F, FMS5E5);
	Get_TEM(dest_Law);
	cout << "FMS5E5" << endl;

	Mat FME5W5;
	FME5W5 = E5T * W5;
	filter2D(dest_, dest_Law, CV_32F, FME5W5);
	Get_TEM(dest_Law);
	cout << "FME5W5" << endl;

	Mat FMW5E5;
	FMW5E5 = W5T * E5;
	filter2D(dest_, dest_Law, CV_32F, FMW5E5);
	Get_TEM(dest_Law);
	cout << "FMW5E5" << endl;

	Mat FME5R5;
	FME5R5 = E5T * R5;
	filter2D(dest_, dest_Law, CV_32F, FME5R5);
	Get_TEM(dest_Law);
	cout << "FME5R5" << endl;

	Mat FMR5E5;
	FMR5E5 = R5T * E5;
	filter2D(dest_, dest_Law, CV_32F, FMR5E5);
	Get_TEM(dest_Law);
	cout << "FMR5E5" << endl;

	Mat FMS5S5;
	FMS5S5 = S5T * S5;
	filter2D(dest_, dest_Law, CV_32F, FMS5S5);
	Get_TEM(dest_Law);
	cout << "FMS5S5" << endl;

	Mat FMS5W5;
	FMS5W5 = S5T * W5;
	filter2D(dest_, dest_Law, CV_32F, FMS5W5);
	Get_TEM(dest_Law);
	cout << "FMS5W5" << endl;

	Mat FMW5W5;
	FMW5W5 = W5T * W5;
	filter2D(dest_, dest_Law, CV_32F, FMW5W5);
	Get_TEM(dest_Law);
	cout << "FMW5W5" << endl;

	Mat FMW5S5;
	FMW5S5 = W5T * S5;
	filter2D(dest_, dest_Law, CV_32F, FMW5S5);
	Get_TEM(dest_Law);
	cout << "FMW5S5" << endl;

	Mat FMS5R5;
	FMS5R5 = S5T * R5;
	filter2D(dest_, dest_Law, CV_32F, FMS5R5);
	Get_TEM(dest_Law);
	cout << "FMS5R5" << endl;

	Mat FMR5R5;
	FMR5R5 = R5T * R5;
	filter2D(dest_, dest_Law, CV_32F, FMR5R5);
	Get_TEM(dest_Law);
	cout << "FMR5R5" << endl;

	Mat FMR5S5;
	FMR5S5 = R5T * S5;
	filter2D(dest_, dest_Law, CV_32F, FMR5S5);
	Get_TEM(dest_Law);
	cout << "FMR5S5" << endl;

	Mat FMW5R5;
	FMW5R5 = W5T * R5;
	filter2D(dest_, dest_Law, CV_32F, FMW5R5);
	Get_TEM(dest_Law);
	cout << "FMW5R5" << endl;

	Mat FMR5W5;
	FMR5W5 = R5T * W5;
	filter2D(dest_, dest_Law, CV_32F, FMR5W5);
	Get_TEM(dest_Law);
	cout << "FMR5W5" << endl;
}

void Feat_Extract::Get_TEM(Mat dest_Law){

	Mat TEM_M;
	Mat TEM_ABSM;
	Mat TEM_SD;
	double size;
	Scalar temp_SD;
	Scalar temp_Mean;
	Scalar ABSM;
	Scalar MS;
	Scalar Entropy;

	//cout << endl << "ABSM: " << endl << endl;
	//-----------------------------
	//-----------------------------
	//-----------------------------
	blur(abs(dest_Law), TEM_ABSM, Size(5, 5));
	normalize(TEM_ABSM, TEM_ABSM, 0, 1, CV_MINMAX);
	imshow("TEM_ABSM", TEM_ABSM);
	//-----------------------------
	ABSM = sum(sum(abs(TEM_ABSM)));
	size = (TEM_ABSM.rows*TEM_ABSM.cols);
	ABSM = ABSM / size;
	ABSM_ABSM = ABSM;

	//-----------------------------
	Mat ABSM_temp;
	log(TEM_ABSM, ABSM_temp);
	multiply(TEM_ABSM,(-ABSM_temp), ABSM_temp);
	Entropy = sum(sum(ABSM_temp));
	Entropy = Entropy / size;
	Entropy_ABSM = Entropy;

	//-----------------------------
	pow(TEM_ABSM, 2, TEM_ABSM);
	MS = sum(sum(TEM_ABSM));
	MS = MS / size;
	MS_ABSM = MS;

	//-----------------------------
	//-----------------------------
	//-----------------------------

	//cout << endl << "MEAN: " << endl << endl;

	//-----------------------------
	//-----------------------------
	//-----------------------------
	blur((dest_Law), TEM_M, Size(5, 5));
	normalize(TEM_M, TEM_M, 0, 1, CV_MINMAX);
	imshow("TEM_M", TEM_M);

	//-----------------------------
	ABSM = sum(sum(abs(TEM_M)));
	size = (TEM_M.rows*TEM_M.cols);
	ABSM = ABSM / size;
	ABSM_Mean = ABSM;

	//-----------------------------
	Mat M_temp;
	log(TEM_M, M_temp);
	multiply(TEM_M, (-M_temp), M_temp);
	Entropy = sum(sum(M_temp));
	Entropy = Entropy / size;
	Entropy_Mean = Entropy;

	//-----------------------------
	pow(TEM_M, 2, TEM_M);
	MS = sum(sum(TEM_M));
	MS = MS / size;
	MS_Mean = MS;



	//-----------------------------
	//-----------------------------
	//-----------------------------


	//cout << endl << "SD: " << endl << endl;
	//-----------------------------
	//-----------------------------
	//-----------------------------
	meanStdDev(dest_Law, temp_Mean, temp_SD);
	Mat temp_val;

	temp_val = dest_Law - temp_Mean[0];
	pow(temp_val, 2, temp_val);
	blur((temp_val), TEM_SD, Size(5, 5));
	sqrt(TEM_SD, TEM_SD);
	normalize(TEM_SD, TEM_SD, 0, 1, CV_MINMAX);
	imshow("TEM_SD", TEM_SD);
	//-----------------------------
	ABSM = sum(sum(abs(TEM_SD)));
	size = (TEM_SD.rows*TEM_SD.cols);
	ABSM = ABSM / size;
	ABSM_SD = ABSM;

	//-----------------------------
	Mat SD_temp;
	log(TEM_SD, SD_temp);
	multiply(TEM_SD, (-SD_temp), SD_temp);
	Entropy = sum(sum(SD_temp));
	Entropy = Entropy / size;
	Entropy_SD = Entropy;

	//-----------------------------
	pow(TEM_SD, 2, TEM_SD);
	MS = sum(sum(TEM_SD));
	MS = MS / size;
	MS_SD = MS;


	//-----------------------------
	//-----------------------------
	//-----------------------------

	//waitKey(0);
	feat_form(&feat_mat);
}


void Feat_Extract::feat_form(Mat *feat_mat){

		feat_vec->push_back(ABSM_ABSM[0]);
		cout << "ABSM_ABSM:     " << ABSM_ABSM[0] << endl;
		feat_vec->push_back(Entropy_ABSM[0]);
		cout << "Entropy_ABSM:  " << Entropy_ABSM[0] << endl;
		feat_vec->push_back(MS_ABSM[0]);
		cout << "MS_ABSM:       " << MS_ABSM[0] << endl;

		feat_vec->push_back(ABSM_Mean[0]);
		cout << "ABSM_Mean:     " << ABSM_Mean[0] << endl;
		feat_vec->push_back(Entropy_Mean[0]);
		cout << "Entropy_Mean:  " << Entropy_Mean[0] << endl;
		feat_vec->push_back(MS_Mean[0]);
		cout << "MS_Mean:       " << MS_Mean[0] << endl;

		feat_vec->push_back(ABSM_SD[0]);
		cout << "ABSM_SD:       " << ABSM_SD[0] << endl;
		feat_vec->push_back(Entropy_SD[0]);
		cout << "Entropy_SD:    " << Entropy_SD[0] << endl;
		feat_vec->push_back(MS_SD[0]);
		cout << "MS_SD:         " << MS_SD[0] << endl;
		cout << endl;

	memcpy(feat_mat->data, FEAT.data(), FEAT.size()*sizeof(CV_32FC1));
}