
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/plot.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/ml/ml.hpp"

#include "fuzzy_clustering.hpp"
#include "Feat_Extract/Feat_Extract.hpp"
#include "GaborFilter/GaborFilter.hpp"
#include "SVM_Set_Train/SVM_Training.hpp"
#include "Wavelet_decomp/Wavelet.hpp"

#include <iostream>
#include <fstream>
#include <iterator>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Ptr<SVM> svm;
//Ptr<SVM> trainSVM(Mat Feature_Vector_);

float siz = 1;

void read_im(Mat *Input, Mat *Input2, String FileName);
void AdaptiveBilateralFilter(Mat Input, Mat *Output, int Siz, int std);
void gray_conv(Mat Input, Mat *Output);
void Clahe_im(Mat Input, Mat *Output);
void PCA_func(Mat feat_vec);
void write_file(String filename, Mat vec);

int main(int argc, char** argv)
{
	Mat Inimage, Inimage2;
	Mat BLF_Image;
	Mat THR;
	Mat Gray_IM;
	Mat Clahe_IM;
	vector<Mat> Gabor;
	vector<Mat> Gabor_ph;
	Mat Feature_Vector(9, 1, CV_32F);

	Mat Final_gab;
	Mat feat_vec;

	float e1, e2, t;
	e1 = getTickCount();

	// READ IMAGE-----------------------------------------


	for (int p = 0; p < 50; p++){

		ostringstream File;
		File << "Images/Samples/Epidermis/Stratum Corneum/" << p+1 << ".PNG";
		//File << "Images/282.tif";
		read_im(&Inimage, &Inimage2, File.str());
		AdaptiveBilateralFilter(Inimage, &BLF_Image, 3, 9);
		gray_conv(BLF_Image, &Gray_IM);
		Clahe_im(Gray_IM, &Clahe_IM);


		Clahe_IM.convertTo(Clahe_IM, CV_32FC1);

		//GaborFilter GaborFilt(Clahe_IM);
		//Gabor = GaborFilt.get_Gab_Ouput();
		//	Gabor_ph = GaborFilt.get_Gab_Ouput_ph();
		//Final_gab = GaborFilt.get_Gab_Final_ph();
		//imshow("final gabor", Final_gab);
		//waitKey(0);

		//Wavelet Wavelet_dec(Final_gab, 1);
		//Mat wavel = Wavelet_dec.get_Wavelet();
		//cout << "wavel: " << endl << wavel << endl;

		//double M = 0, m = 0;
		//----------------------------------------------------
		// Normalization to 0-1 range (for visualization)
		//----------------------------------------------------
		//minMaxLoc(wavel, &m, &M);
		//if ((M - m)>0) { wavel = wavel*(255.0 / (M - m)) - m / (M - m); }

		//cout << "wavel: " << endl << wavel << endl;

		//imshow("wavel", wavel);
		//waitKey(0);



		Feat_Extract Feat(Clahe_IM);
		feat_vec = Feat.get_feat_mat();

		

		
		//cout << "feat feat_vec: " << endl << feat_vec << endl;
		
		

		//hconcat(Feature_Vector, feat_vec, Feature_Vector);



	}
	imshow("pause", Inimage);
	cout << "p: " << endl << feat_vec << endl;
	waitKey(0);

	Mat Inimage1, Inimage3;
	Mat BLF_Image1;
	Mat THR1;
	Mat Gray_IM1;
	Mat Clahe_IM1;
	vector<Mat> Gabor1;
	vector<Mat> Gabor_ph1;
	Mat Feature_Vector1(9, 1, CV_32F);

	Mat Final_gab1;
	Mat feat_vec1;

	for (int p = 0; p < 50; p++){

		ostringstream File1;
		File1 << "Images/Samples/Epidermis/SG_SP_SB/" << p + 1 << ".PNG";

		read_im(&Inimage1, &Inimage3, File1.str());
		AdaptiveBilateralFilter(Inimage1, &BLF_Image1, 3, 9);
		gray_conv(BLF_Image1, &Gray_IM1);
		Clahe_im(Gray_IM1, &Clahe_IM1);

		Clahe_IM1.convertTo(Clahe_IM1, CV_32FC1);


		//Perfoming Gabor with Clahe
		

		//GaborFilter GaborFilt1(Clahe_IM1);
		//Gabor1 = GaborFilt1.get_Gab_Ouput();
		//Gabor_ph1 = GaborFilt1.get_Gab_Ouput_ph();
		//Final_gab1 = GaborFilt1.get_Gab_Final_ph();
		//imshow("Final_gab1", Final_gab1);
		//waitKey(0);






		//Perfoming Wavelet with Clahe


		//Wavelet Wavelet_dec(Final_gab1, 1);
		//Mat wavel = Wavelet_dec.get_Wavelet();
		//cout << "wavel: " << endl << wavel << endl;

		//double M = 0, m = 0;
		//----------------------------------------------------
		// Normalization to 0-1 range (for visualization)
		//----------------------------------------------------
		//minMaxLoc(wavel, &m, &M);
		//if ((M - m)>0) { wavel = wavel*(255.0 / (M - m)) - m / (M - m); }







	

		




		Feat_Extract Feat1(Clahe_IM1);
		feat_vec1 = Feat1.get_feat_mat();
		
		//hconcat(Feature_Vector1, feat_vec1, Feature_Vector1);
		cout << "p: " << endl << p << endl;
		//cout << "feat feat_vec: " << endl << feat_vec1 << endl;
	}

	imshow("pause", Inimage);
	cout << "p: " << endl << feat_vec << endl;
	waitKey(0);

	Mat Inimage4, Inimage5;
	Mat BLF_Image2;
	Mat THR2;
	Mat Gray_IM2;
	Mat Clahe_IM2;
	vector<Mat> Gabor2;
	vector<Mat> Gabor_ph2;
	Mat Feature_Vector2(9, 1, CV_32F);

	Mat Final_gab2;
	Mat feat_vec2;

	for (int p = 0; p < 30; p++){

		ostringstream File2;
		File2 << "Images/Samples/Dermis/" << p + 1 << ".PNG";

		read_im(&Inimage4, &Inimage5, File2.str());
		AdaptiveBilateralFilter(Inimage4, &BLF_Image2, 3, 9);
		gray_conv(BLF_Image2, &Gray_IM2);
		Clahe_im(Gray_IM2, &Clahe_IM2);


		Clahe_IM2.convertTo(Clahe_IM2, CV_32FC1);





		//GaborFilter GaborFilt2(Clahe_IM2);
		//Gabor2 = GaborFilt2.get_Gab_Ouput();
		//Gabor_ph2 = GaborFilt2.get_Gab_Ouput_ph();
		//Final_gab2 = GaborFilt2.get_Gab_Final_ph();
		//imshow("Final_gab2", Final_gab2);
		//waitKey(0);





		//Wavelet Wavelet_dec(Final_gab2, 1);
		//Mat wavel = Wavelet_dec.get_Wavelet();
		//cout << "wavel: " << endl << wavel << endl;

		//double M = 0, m = 0;
		//----------------------------------------------------
		// Normalization to 0-1 range (for visualization)
		//----------------------------------------------------
		//minMaxLoc(wavel, &m, &M);
		//if ((M - m)>0) { wavel = wavel*(255.0 / (M - m)) - m / (M - m); }








		 
		Feat_Extract Feat2(Clahe_IM2);
		feat_vec2 = Feat2.get_feat_mat();

		//hconcat(Feature_Vector2, feat_vec2, Feature_Vector2);
		cout << "p: " << endl << p << endl;
		//cout << "feat feat_vec: " << endl << feat_vec2 << endl;

	}

	imshow("pause", Inimage);
	waitKey(0);
	
	Mat Feature_Vector_;
	Mat Feature_Vector_1;
	Mat Feature_Vector_2;
	Feature_Vector.colRange(1, 51).copyTo(Feature_Vector_);
	Feature_Vector1.colRange(1, 51).copyTo(Feature_Vector_1);
	Feature_Vector2.colRange(1, 31).copyTo(Feature_Vector_2);

	hconcat(Feature_Vector_, Feature_Vector_1, Feature_Vector_);
	hconcat(Feature_Vector_, Feature_Vector_2, Feature_Vector_);
	
	PCA_func(Feature_Vector_);

	write_file("FeatVec/Feature_Vector.csv", Feature_Vector_);

	Mat trainmat;
	Mat labelsmat;
	
	Ptr<SVM> svm_;

	e2 = getTickCount();
	t = (e2 - e1) / getTickFrequency();
	cout << "time to run: " << t << endl;

	//svm_ = trainSVM(Feature_Vector_);

	//SVM_Training SVM(Feature_Vector_);
	//svm = SVM.get_svm_trained();

	
	e2 = getTickCount();
	t = (e2 - e1) / getTickFrequency();
	cout << "time to run: " << t << endl;



	waitKey(0);
	return 0;
}

void write_file(String filename, Mat vec){

	ofstream myfile;
	ostringstream filename1;

	filename1 << filename;

	cv::Formatter const * c_formatter(cv::Formatter::get("CSV"));


	myfile.open(filename1.str());
	c_formatter->write(myfile, vec);
	myfile.close();

}

void read_im(Mat *Input, Mat *Input2, String FileName){

	*Input = imread(FileName, CV_LOAD_IMAGE_COLOR);

	Mat temp;
	Size dsize;

}

void AdaptiveBilateralFilter(Mat Input, Mat *Output, int Siz, int std){

	Mat temp;
	Size dsize;

	adaptiveBilateralFilter(Input, *Output, Size(Siz, Siz), std);
}

void gray_conv(Mat Input, Mat *Output){

	cvtColor(Input, *Output, CV_BGR2GRAY);
}

void Clahe_im(Mat Input, Mat *Output){

	Mat temp;
	Size dsize;
	Ptr<CLAHE> clahe = createCLAHE();

	clahe->setClipLimit(1.6);
	clahe->apply(Input, *Output);

	imwrite("IMclahe.tif", *Output);

}

void PCA_func(Mat feat_vec){

	Mat eigenVec;
	Mat projection_result;
	
	Mat back_proj = Mat::zeros(feat_vec.rows, feat_vec.cols, CV_32FC1);
	double Variance;

	PCA pca_analysis(feat_vec, Mat(), CV_PCA_DATA_AS_COL, 4);

	//cout << endl << "PCA EigenValues: " << pca_analysis.eigenvalues << endl;
	//cout << endl << "PCA EigenVectors: " << pca_analysis.eigenvectors << endl;


	pca_analysis.project(feat_vec, projection_result);
	//cout << endl << "PCA projection_result: " << projection_result << endl;

	back_proj = pca_analysis.backProject(projection_result);

	//cout << endl << "PCA back_proj: " << back_proj << endl;


	write_file("FeatVec/PCA_Projection_Result.csv", projection_result);

	write_file("FeatVec/PCA_Eigen_Vectors.csv", pca_analysis.eigenvectors);

	write_file("FeatVec/PCA_Back_Project.csv", back_proj);

}


