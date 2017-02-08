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
	void histog(Mat Input, Mat *hist_graph);
	void GLCM(Mat Input, Mat *glcm);
	void ASM(Mat GLCM, double *ASM_);
	void Correl(Mat GLCM, double *corr_);
	void IDM(Mat GLCM, double *IDM_);
	void feat_form(Mat *feat_mat);

	void Fourier_dft(double *dft_entropy, double *dft_inertia, double *dft_energy);

	void Laws_filter();
	void Get_TEM(Mat dest_law);





	Mat get_feat_mat() { return feat_mat; }



private:

	Mat feat_mat = Mat(9, 50, CV_32FC1);

	//double Gabor_mean;
	//double Gabor_std;	
	//double Gabor_energy;	
	//double Gabor_entropy;
	//double glcm_entropy;
	//Mat hist_graph;
	//Mat hist_graph_ph;
	//double ASM_;
	//double corr_;
	//double IDM_;
	//Mat glcm;

	Mat dest_;
	vector<Mat> dest_ph_;

	vector<float> FEAT;
	vector<float> *feat_vec = &FEAT;
	double temp;


	Scalar ABSM_ABSM;
	Scalar MS_ABSM;
	Scalar Entropy_ABSM;

	Scalar ABSM_Mean;
	Scalar MS_Mean;
	Scalar Entropy_Mean;

	Scalar ABSM_SD;
	Scalar MS_SD;
	Scalar Entropy_SD;

	//double dft_entropy;
	//double dft_inertia;
	//double dft_energy;




};