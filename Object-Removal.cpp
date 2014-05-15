#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <stdio.h>


using namespace std;
using namespace cv;

void drawRectangles(vector<Point> points, vector<Rect>& rects){
	int x_min=1000000000, x_max=0, y_min=1000000000, y_max=0;
	for(int i=0;i<points.size();i++){
		if(x_min>points[i].x)
			x_min = points[i].x;
		if(x_max<points[i].x)
			x_max = points[i].x;
		if(y_min>points[i].y)
			y_min = points[i].y;
		if(y_max<points[i].y)
			y_max = points[i].y;
	}
	if(((x_max-x_min) * (y_max-y_min))>10){
		rects.push_back(Rect(x_min, y_min, (x_max-x_min), (y_max-y_min)));
	}
}


bool frame_ok(Mat image){
	int count = 0;
	
	for(int i=0;i<image.rows;i++){
		for(int j=0;j<image.cols;j++){
			count +=(image.data[image.channels()*(image.cols*i + j) + 1])>0?1:0;
		}
	}
	cout<<count<<endl;
	if(count<0.20*(image.rows*image.cols))
		return true;

	return false;
}

bool frame_similar(Mat x, Mat y){

	int h_bins = 50;
	int histSize[] = { h_bins };
	float hranges[] = { 0, 255 };
	
	const float* ranges[] = { hranges };

	MatND hist_x;
	MatND hist_y;
	int channels[] = { 0 };

	calcHist( &x, 1, channels, Mat(), hist_x, 1, histSize, ranges, true, false );
	normalize( hist_x, hist_x, 0, 1, NORM_MINMAX, -1, Mat() );

	calcHist( &y, 1, channels, Mat(), hist_y, 1, histSize, ranges, true, false );
	normalize( hist_y, hist_y, 0, 1, NORM_MINMAX, -1, Mat() );

	double value = compareHist( hist_x, hist_y, 0 );

	if(value > 0.75)
		return true;
	return false;
}

void mergerimages(Mat a, Mat& b, Rect r){
	for(int i=r.y;i<(r.y+r.height);i++){
		for(int j=r.x;j<(r.x+r.width);j++){
			b.data[b.channels()*(b.cols*i + j) + 0]=(a.data[a.channels()*(a.cols*i + j) + 0]);
			b.data[b.channels()*(b.cols*i + j) + 1]=(a.data[a.channels()*(a.cols*i + j) + 1]);
			b.data[b.channels()*(b.cols*i + j) + 2]=(a.data[a.channels()*(a.cols*i + j) + 2]);
		}
	}
}


int main( int argc, char** argv ){
	Mat image_A = imread( argv[1]);
	Mat image_B = imread( argv[2]);
	Mat image_C = imread( argv[2]);
	
	Mat image1, image2, image3, image4;
	cvtColor(image_A,image1,CV_RGB2GRAY);
	cvtColor(image_B,image2,CV_RGB2GRAY);
	std::vector<std::vector<cv::Point> > contours;
	vector<Rect> rectangles;
	absdiff(image2, image1, image3);
	threshold( image3, image4, 0, 255, 0 );
	cv::erode(image4,image4,cv::Mat());
        cv::dilate(image4,image4,cv::Mat());

	findContours(image4,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	for(int i=0;i<contours.size();i++){
		drawRectangles(contours[i], rectangles);
	}
	drawContours(image3,contours,-1,cv::Scalar(255,255,255),-1);
	rectangle( image3, cvPoint(rectangles[0].x, rectangles[0].y),cvPoint(rectangles[0].x+rectangles[0].width, rectangles[0].y+rectangles[0].height), CV_RGB(255,255,255), 3, 8, 0);
rectangle( image3, cvPoint(rectangles[1].x, rectangles[1].y),cvPoint(rectangles[1].x+rectangles[1].width, rectangles[1].y+rectangles[1].height), CV_RGB(255,255,255), 3, 8, 0);


	Mat img1a,img1b,img2a,img2b;
	img1a = image1(rectangles[0]);
	img1b = image1(rectangles[1]);
	img2a = image2(rectangles[0]);
	img2b = image2(rectangles[1]);

	Mat temp;
	if(frame_similar(img1a, img2b)){
		mergerimages(image_A, image_C, rectangles[1]);
		mergerimages(image_B, image_C, rectangles[0]);
	}
	else if(frame_similar(img1b, img2a)){
		mergerimages(image_A, image_C, rectangles[0]);
		mergerimages(image_B, image_C, rectangles[1]);

	}

	imwrite("result.jpg", image_C);
	imwrite("img1a.jpg", img1a);
	imwrite("img1b.jpg", img1b);
	imwrite("img2a.jpg", img2a);
	imwrite("img2b.jpg", img2b);
	imwrite("result-a.jpg", image1);
	imwrite("result-b.jpg", image2);
	imwrite("result-c.jpg", image3);
	return 0;
}
