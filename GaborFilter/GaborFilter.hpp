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

class GaborFilter{

public:

	GaborFilter(Mat Input) :
		input_(Input)
	{
		initEverything();
	};


	void initEverything();
	void get_kernel(vector<Mat>* Kernel, vector<Mat>* Kernel_ph, Mat Clahe_IM);
	Mat GaborFilt(double pos_th, double Sigma, double Lambda, double Gamma, double psi, int i, int ks);

	vector<Mat> get_Gab_Ouput() { return Gabor; }
	vector<Mat> get_Gab_Ouput_ph() { return Gabor_ph; }
	Mat get_Gab_Final_ph() { return Final_gabor; }

private:
	
	Mat input_;
	vector<Mat> Gabor;
	vector<Mat> Gabor_ph;
	Mat Final_gabor;

};