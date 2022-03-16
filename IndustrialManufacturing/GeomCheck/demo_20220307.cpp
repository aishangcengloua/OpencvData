#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;

bool FlatternCircle(const Mat& inputImg, OutputArray outputImg, 
	const Point& center, const int nRadius, const int nRingHeight)
{
	if (inputImg.empty())
		return false;

	//输出图像为矩形，高度为圆环高度，宽度为圆环外圆周长
	outputImg.create(Size(nRadius*CV_2PI, nRingHeight), CV_8UC1);
	Mat rectangle = outputImg.getMat();
	int rows = rectangle.rows;
	int cols = rectangle.cols;

	for (int j = 0; j < rows; j++)
	{
		uchar* data = rectangle.ptr<uchar>(j);
		for (int i = 0; i < cols; i++)
		{
			//根据极坐标计算公式设置展平矩形对应位置的像素值
			double theta = CV_2PI / float(cols) * float(i + 1);
			double rho = nRadius - j - 1;
			int x = (float)center.x + rho*std::cos(theta) + 0.5;
			int y = (float)center.y + rho*std::sin(theta) + 0.5;
			if(y < inputImg.rows && x < inputImg.cols)
				data[i] = inputImg.at<uchar>(y, x);
		}
	}

	return true;
}

int main()
{    
	//读取彩色图像
	std::string strImgFile = "E:\\SZU\\Spring2022\\MachineVision\\PPT\\demos\\images\\circle_band.bmp";
	Mat img = imread(strImgFile);
	if (img.empty())
	{    
		std::cout << "image file read failed - " << strImgFile.c_str() << "!" << std::endl;
		return 1;
	}

	//转换灰度图
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 1.0);
	//threshold(gray, gray, 100, 255, cv::THRESH_BINARY);// not helping

	Mat edge;
	cv::Canny(gray, edge, 120, 255);

	//检测最大的圆
	std::vector<Vec3f> circles;
	HoughCircles(edge, circles, HOUGH_GRADIENT,
		1,		// dp 
		5,		// minDist
		240		// param1
		);

	Mat img_copy;
	img.copyTo(img_copy);
	int nMaxRadius = 0;
	Point center;
	for (int i = 0; i < circles.size(); i++){
		Vec3f cc = circles[i];
		if (nMaxRadius < cc[2]) {
			nMaxRadius = int(cc[2] + 0.5);
			center = Point(cc[0], cc[1]);
		}
		circle(img_copy, Point(cc[0], cc[1]), cc[2], Scalar(0, 0, 255), 2, LINE_AA);
		circle(img_copy, Point(cc[0], cc[1]), 3, Scalar(125, 25, 255), 2, LINE_AA);
	}

	imshow("circles", img_copy);
	waitKey(0);

	//将最大的圆环展平，圆环高度为半径的一半
	Mat result;
	FlatternCircle(gray, result, center, nMaxRadius, nMaxRadius / 2);
	if (result.empty()) {
		std::cout << "result is empty!" << std::endl;
		return 1;
	}
	else {
		imshow("origin", img);
		imshow("result", result);
		waitKey(0);
		destroyAllWindows();
	}

	return 0;
}