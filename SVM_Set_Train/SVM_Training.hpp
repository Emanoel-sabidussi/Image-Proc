#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/plot.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace std;

class SVM_Training {
public:

	SVM_Training(Mat dest) :
		dest_(dest)

	{
		initEverything();
	};

	void initEverything();
	void trainSVM();
	Ptr<SVM> get_svm_trained() { return svm;}

	void SVMpredict(Mat feat_vec);
	void draw_svm();


private:

	Mat dest_;
	Ptr<SVM> svm;

	Mat test;
	CvSVMParams params;
	CvSVM SVM;

	int width = 1024, heigth = 1024;
	Mat image = Mat::zeros(heigth, width, CV_8UC3);

};