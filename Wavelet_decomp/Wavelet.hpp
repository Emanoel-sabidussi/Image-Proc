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

class Wavelet {
public:

	Wavelet(Mat src_,  int NIter_) :
		src(src_),
		NIter(NIter_)

	{
		initEverything();
	};


	void initEverything();
	void Wavelet_decomp();

	float sgn(float x);
	float soft_shrink(float d, float T);
	float hard_shrink(float d, float T);
	float Garrot_shrink(float d, float T);
	void cvHaarWavelet();



	Mat get_Wavelet() { return dst; }

private:

	Mat dst;

	Mat src;
	string name;
	int NIter;

	


};