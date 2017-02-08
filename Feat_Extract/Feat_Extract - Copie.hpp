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

using namespace cv;
using namespace std;

class Feat_Extract {
public:



	Feat_Extract(Mat dest):
		dest_(dest)
	{
		initEverything();
	};
	

	void initEverything();
	
	void std_mean(int i, double *Gabor_mean, double *Gabor_std);
	void energy(int i, double *Gabor_energy);
	void entropy(Mat dest, double *Gabor_entropy, int flag);
	void histog(Mat Input, vector<Mat> *hist_graph);
	void GLCM(Mat Input, Mat *glcm);
	void ASM(Mat GLCM, double *ASM_);
	void Correl(Mat GLCM, double *corr_);
	void IDM(Mat GLCM, double *IDM_);
	void feat_form(Mat *feat_mat);

	Mat get_feat_mat() { return feat_mat; }



private:

	Mat feat_mat = Mat(8, 1, CV_32FC1);

	double Gabor_mean;
	double Gabor_std;	
	double Gabor_energy;	
	double Gabor_entropy;
	double glcm_entropy;
	vector<Mat> hist_graph;
	vector<Mat> hist_graph_ph;
	double ASM_;
	double corr_;
	double IDM_;
	Mat glcm;

	Mat dest_;
	vector<Mat> dest_ph_;

	vector<float> FEAT;
	vector<float> *feat_vec = &FEAT;
	double temp;
};