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
#include <time.h>

using namespace std;
using namespace cv;

vector<Rect> collection_rectangles;

int object_counter = 0;

class ObjectInfo{
	public:
		Rect rectangle;
		int counter;
		
		ObjectInfo(Rect rect, int cnt){
			setRectangle(rect);
			setCounter(cnt);

		}
		void setRectangle(Rect rect){
			rectangle = rect;
		}
		void setCounter(int cnt){
			counter = cnt;
		}

};


vector<ObjectInfo> final_rectangles;


int distance(cv::Point p1, cv::Point p2){
	int x = p2.x - p1.x;
	int y = p2.y - p1.y;
	return sqrt(x*x + y*y);
}


bool valueInRange(int value, int min, int max)
{ return (value >= min) && (value <= max); }


bool intersect(vector<int> x, vector< vector< int> > y){

	for(int i=0;i<y.size(); i++){
		vector<int> vec = y[i];
		bool xoverlap = valueInRange(x[1], vec[1], vec[3]) || valueInRange(vec[1], x[1], x[3]);
		bool yoverlap = valueInRange(x[2], vec[2], vec[4]) || valueInRange(vec[2], x[2], x[4]);
		
		if(xoverlap && yoverlap){
			return true;
		}
	}
	return false;
}

bool rectOverlap(vector<Rect> avail_faces, Rect r){
	for(int i=0;i<avail_faces.size(); i++){
		Rect rectangle = avail_faces[i];
		bool xoverlap = valueInRange(r.x, rectangle.x, rectangle.x + rectangle.width) || 
				valueInRange(rectangle.x, r.x, r.x + r.width);
		bool yoverlap = valueInRange(r.y, rectangle.y, rectangle.y + rectangle.height) || 
				valueInRange(rectangle.y, r.y, r.y + r.height);
		
		if(xoverlap && yoverlap){
			return true;
		}
	}
	return false;
}



bool frame_ok(Mat frame, Rect r){
	Mat image(r.width, r.height, CV_8UC1);
	image = frame(r);
	int count = 0;
	vector<Rect> faces;

	
	int mx = 0;
	for(int i=0;i<image.rows;i++){
		for(int j=0;j<image.cols;j++){
			mx = max(mx,(int)image.data[image.channels()*(image.cols*i + j) + 1]);
			count +=(image.data[image.channels()*(image.cols*i + j) + 1]/130);
		}
	}
	
	if(count>15)
		return true;

	return false;
}




void drawRectangles(Mat& frame,Mat fore, vector<vector<Point> >& vec){
	vector<Rect> rectangles;
	for(int i=0;i<vec.size();i++){
		vector<Point> temp = vec[i];
		int left_x, left_y, right_x, right_y;
		left_x = right_x = temp[0].x;
		left_y = right_y = temp[0].y;
		for(int j=temp.size()>1?1:0;j<vec[i].size();j++){
			if(left_x > temp[j].x) left_x = temp[j].x;
			if(right_x < temp[j].x) right_x = temp[j].x;
			if(left_y > temp[j].y) left_y = temp[j].y;
			if(right_y < temp[j].y) right_y = temp[j].y;
		}
		if((((right_x - left_x)*(right_y - left_y))>2000) && (frame_ok(fore, Rect(left_x, left_y, (right_x - left_x), (right_y - left_y))))){
			rectangles.push_back(Rect(left_x, left_y, (right_x - left_x), (right_y - left_y)));
			rectangle( frame, cvPoint(left_x, left_y), cvPoint(right_x, right_y), CV_RGB(0,255,0), 3, 8, 0);
			
		}
	}
	if(rectangles.size()>collection_rectangles.size()){
		int counter = 0;
		for(int i=0;i<collection_rectangles.size();i++){
			Rect r1 = collection_rectangles[i];
			Rect r2 = rectangles[0];
			int index = 0;
			cv::Point p1 = Point((r1.x+r1.width/2),(r1.y+r1.height/2));
			cv::Point p2 = Point((r2.x+r2.width/2),(r2.y+r2.height/2));
			int x = p2.x - p1.x;
			int y = p2.y - p1.y;
			
			int dist = sqrt(x*x + y*y);
			for(int j=1;j<rectangles.size();j++){
				Rect r3 = rectangles[j];
				Point p3 = cvPoint((r3.x+r3.width/2),(r3.y+r3.height/2));
				int x_ = p3.x - p1.x;
				int y_ = p3.y - p1.y;
				
				int dist_2 = sqrt(x_*x_ + y_*y_);
				if(dist_2<dist){
					index = j;
					dist = dist_2;
				}
			}
			
			if(dist>10){
				Rect temp_rect = Rect(rectangles[index].x, rectangles[index].y, rectangles[index].width, rectangles[index].height);
				collection_rectangles[i].x = rectangles[index].x;
				collection_rectangles[i].y = rectangles[index].y;
				collection_rectangles[i].width = rectangles[index].width;
				collection_rectangles[i].height = rectangles[index].height;

				ObjectInfo obif = final_rectangles[i];
				
				final_rectangles[i].rectangle = temp_rect;
			}
			rectangles.erase(rectangles.begin()+index);
			counter++;
			
		}
		for(int i=0;i<rectangles.size();i++){
			if(!rectOverlap(collection_rectangles, rectangles[i])){
				collection_rectangles.push_back(rectangles[i]);
				ObjectInfo obj(rectangles[i], object_counter);
				final_rectangles.push_back(obj);
				object_counter++;
			}
			
		}
	}
	else{
		int counter = 0;
		for(int i=0;i<collection_rectangles.size();i++){
			Rect r1 = collection_rectangles[i];
			if(rectangles.size()>0){
				Rect r2 = rectangles[0];
				int index = 0;
				cv::Point p1 = Point((r1.x+r1.width/2),(r1.y+r1.height/2));
				cv::Point p2 = Point((r2.x+r2.width/2),(r2.y+r2.height/2));
				int x = p2.x - p1.x;
				int y = p2.y - p1.y;
				
				int dist = sqrt(x*x + y*y);
				for(int j=1;j<rectangles.size();j++){
					Rect r3 = rectangles[j];
					Point p3 = cvPoint((r3.x+r3.width/2),(r3.y+r3.height/2));
					int x_ = p3.x - p1.x;
					int y_ = p3.y - p1.y;
					int dist_2 = sqrt(x_*x_ + y_*y_);
					if(dist_2<dist){
						index = j;
						dist = dist_2;
					}
				}
				
				if(dist>10){
					collection_rectangles[i].x = rectangles[index].x;
					collection_rectangles[i].y = rectangles[index].y;
					collection_rectangles[i].width = rectangles[index].width;
					collection_rectangles[i].height = rectangles[index].height;

					Rect temp_rect = Rect(rectangles[index].x, rectangles[index].y, rectangles[index].width, 
rectangles[index].height);
					ObjectInfo obif = final_rectangles[i];
					
					final_rectangles[i].rectangle = temp_rect;
					

				}
				rectangles.erase(rectangles.begin()+index);
				counter++;
			}
		}
		
	}
	
	for(int i=0;i<collection_rectangles.size();i++){
		ObjectInfo obif = final_rectangles[i];
		if(!frame_ok(fore, collection_rectangles[i])){
			
			collection_rectangles.erase(collection_rectangles.begin()+i);
			final_rectangles.erase(final_rectangles.begin()+i);
			--i;	
			
		}
	}
	

	for(int i=0;i<final_rectangles.size();i++){
		ObjectInfo obif = final_rectangles[i];
		
		putText(frame, format("%d",obif.counter), cvPoint(final_rectangles[i].rectangle.x, final_rectangles[i].rectangle.y-2), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0,0,0), 1, CV_AA);
		rectangle( frame, cvPoint(final_rectangles[i].rectangle.x, final_rectangles[i].rectangle.y),
		 cvPoint(final_rectangles[i].rectangle.x + final_rectangles[i].rectangle.width, final_rectangles[i].rectangle.y + final_rectangles[i].rectangle.height), CV_RGB(0,0,0), 3, 8, 0);

	}
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
	
	int x_start, x_end, y_start, y_end;

	if(r.y>=30)
		y_start = r.y-30;
	else
		y_start = 0;
	if((r.y+r.height+30) <= a.rows)
		y_end = (r.y+r.height+30);
	else
		y_end = a.rows;

	if(r.x>=30)
		x_start = r.x-30;
	else
		x_start = 0;
	if((r.x+r.width+30) <= a.cols)
		x_end = (r.x+r.width+30);
	else
		x_end = a.cols;




	for(int i=y_start;i<y_end;i++){
		for(int j=x_start;j<x_end;j++){
			b.data[b.channels()*(b.cols*i + j) + 0]=(a.data[a.channels()*(a.cols*i + j) + 0]);
			b.data[b.channels()*(b.cols*i + j) + 1]=(a.data[a.channels()*(a.cols*i + j) + 1]);
			b.data[b.channels()*(b.cols*i + j) + 2]=(a.data[a.channels()*(a.cols*i + j) + 2]);
		}
	}
	
}


int main(int argc, char *argv[])
{
    cv::Mat frame;
    cv::Mat back;
    cv::Mat fore;
    cv::Mat main_frame;
    cv::Mat first_frame, last_frame;
    cv::VideoCapture cap(0);
    cv::BackgroundSubtractorMOG2 bg(1000, 300, false);

    std::vector<std::vector<cv::Point> > contours;

    cap >> main_frame;
    for(;;)
    {
        cap >> frame;
	if(!first_frame.data){
		frame.copyTo(first_frame);
	}
	else{
		frame.copyTo(last_frame);
	}
	Mat yuv;

	cv::cvtColor(frame, frame, CV_BGR2Lab);

        bg.operator ()(frame,fore,0.000001);

        cv::erode(fore,fore,cv::Mat());
        cv::dilate(fore,fore,cv::Mat());
	cv::imshow("Foreground", fore);
        cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	std::vector<std::vector<cv::Point> > final_contours;
	for(int i=0;i<contours.size();i++){
		//if(contours[i].size()>300 && contours[i].size()<2000){
		if(contours[i].size()>300){
			final_contours.push_back(contours[i]);
			//cout<<contours[i].size()<<endl;
		}
	}
	drawRectangles(frame, fore.clone(), final_contours);
        cv::drawContours(frame,final_contours,-1,cv::Scalar(0,0,255),-1);
        cv::imshow("Frame",frame);
	cv::cvtColor(frame, frame, CV_Lab2RGB);

        if(cv::waitKey(30) >= 0){
		
		//cv::cvtColor(last_frame, last_frame, CV_Lab2BGR);
		break;
	}
    }
    cv::Mat image1a = first_frame(final_rectangles[0].rectangle);
    cv::Mat image1b = first_frame(final_rectangles[1].rectangle);
    cv::Mat image2a = last_frame(final_rectangles[0].rectangle);
    cv::Mat image2b = last_frame(final_rectangles[1].rectangle);
    cv::Mat final_frame = last_frame.clone();

    cvtColor(image1a,image1a,CV_RGB2GRAY);
    cvtColor(image1b,image1b,CV_RGB2GRAY);
    cvtColor(image2a,image2a,CV_RGB2GRAY);
    cvtColor(image2b,image2b,CV_RGB2GRAY);

    //cout<<"No of rectangle: "<<final_rectangles.size()<<endl;

    if(frame_similar(image1a, image2b)){
		//cout<<"Moved Right"<<endl;
		mergerimages(first_frame, final_frame, final_rectangles[0].rectangle);
		mergerimages(last_frame, final_frame, final_rectangles[1].rectangle);
    }
    else if(frame_similar(image1b, image2a)){
		//cout<<"Moved Left"<<endl;
		mergerimages(first_frame, final_frame, final_rectangles[0].rectangle);
		mergerimages(last_frame, final_frame, final_rectangles[1].rectangle);

    }

    //imwrite("Image1a.jpg", image1a);
    //imwrite("Image1b.jpg", image1b);
    //imwrite("Image2a.jpg", image2a);
    //imwrite("Image2b.jpg", image2b);
    imwrite("Final-Image.jpg", final_frame);

    return 0;
}
