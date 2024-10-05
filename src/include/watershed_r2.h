#pragma once
#include "opencv2/core/core.hpp"

void findObjects(const cv::Mat& image, std::vector <cv::Mat>& objects);

struct WatershedObjectBorder {
	WatershedObjectBorder()
	{
		mask = cv::Mat();
		tl = cv::Point();
		br = cv::Point();
	};
	WatershedObjectBorder(cv::Size sz)
	{
		mask = cv::Mat::zeros(sz, CV_8UC1);
		tl = cv::Point(INT_MAX, INT_MAX);
		br = cv::Point(-1, -1);
	};
	cv::Mat mask;
	cv::Point tl;
	cv::Point br;
};

