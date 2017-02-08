
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/plot.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/ml/ml.hpp"

#include "SVM_Training.hpp"


using namespace std;


void SVM_Training::initEverything() {

	trainSVM();
}

void SVM_Training::trainSVM(){

	// Set up training data

	Mat label1 = Mat::ones(1, 50, CV_32FC1);
	Mat label3 = Mat::ones(1, 30, CV_32FC1);
	Mat labelsMat(1, 80, CV_32FC1);

	label3 = label3*(-1);	
	hconcat(label1, label3, labelsMat);

	Mat dest_1;
	Mat dest_11;
	Mat dest_2;
	Mat dest_22;
	Mat dest_3;
	Mat dest_33;
	
	dest_.row(6).copyTo(dest_1);
	cout << "dest_1.colRange(0, 49): " << dest_.row(6) << endl;
	dest_.row(1).copyTo(dest_11);
	hconcat(dest_1, dest_11, dest_1);

	dest_1.colRange(0, 50).copyTo(dest_1);
	cout << "dest_1.colRange(0, 49): " << dest_1.size() << endl;
	
	dest_.rowRange(6, 8).copyTo(dest_2);
	dest_2.colRange(50, 100).copyTo(dest_2);

	dest_.rowRange(6, 7).copyTo(dest_3);
	dest_.rowRange(1, 2).copyTo(dest_33);
	hconcat(dest_3, dest_33, dest_3);

	dest_3.colRange(100, 130).copyTo(dest_3);
	cout << "dest_3.colRange(100, 130): " << dest_3.size() << endl;
	
	hconcat(dest_1, dest_3, dest_);

	dest_.convertTo(dest_, CV_32FC1);
	transpose(dest_, dest_);

	//dest_ = dest_ * 50;

	Mat trainingDataMat(80, 2, CV_32FC1);

	dest_.copyTo(trainingDataMat);

	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.degree = 4;
	params.gamma = 0.00001;
	params.coef0 = 10;

	//params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, (int)45000000, 1e-10);
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	draw_svm();

}

void SVM_Training::draw_svm(){

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = SVM.predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;
		}


	}


	// Show the training data
	int thickness = -1;
	int lineType = 8;

	for (int i = 0; i < 49; i++){

		circle(image, Point(double(round(dest_.at<float>(i, 0))), double(round(dest_.at<float>(i, 1)))), 5, Scalar(0, 0, 0), thickness, lineType);

	}


	for (int i = 49; i < 78; i++){

		circle(image, Point(double(round(dest_.at<float>(i, 0))), double(round(dest_.at<float>(i, 1)))), 5, Scalar(255, 255, 255), thickness, lineType);

	}


	//circle(image, Point(double(round(dest_.at<float>(44, 0))), double(round(dest_.at<float>(44, 1)))), 5, Scalar(100, 100, 50), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	int c = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
		const float* v = SVM.get_support_vector(i);
		//circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	//Ptr<SVM> svm_;
	


}

void SVM_Training::SVMpredict(Mat feat_vec){

	int i;

	

	feat_vec.rowRange(1, 3).copyTo(feat_vec);


	feat_vec = feat_vec * 200;

	cout << "feat_vec0: " << feat_vec.at<float>(0, 0) << endl;
	cout << "feat_vec1: " << feat_vec.at<float>(0, 1) << endl;

	Mat sampleMat = (Mat_<float>(1, 2) << feat_vec.at<float>(0, 0), feat_vec.at<float>(0, 1));

	cout << "sampleMat: " << sampleMat.at<float>(0, 1) << endl;

	float response = SVM.predict(sampleMat);

	cout << "response: " << response << endl;

	if (response == 1)
		cout << "layer 1" << endl;
	else if (response == -1)
		cout << "layer 2" << endl;

	int thickness = -1;
	int lineType = 8;
	circle(image, Point(double(round(feat_vec.at<float>(0, 0))), double(round(feat_vec.at<float>(0, 1)))), 5, Scalar(0, 255, 255), thickness, lineType);




	imshow("SVM Simple Example", image); // show it to the user


}