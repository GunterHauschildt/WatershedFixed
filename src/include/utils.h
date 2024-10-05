#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <vector>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

///////////////////////////////

class Colors {
public:
	Colors(int l = 64);
	static cv::Vec3b at(int n);
	static cv::Vec3b at();
private:
	static std::unordered_map<int, cv::Vec3b> m_Colors;

};

///////////////////////////////

class NamedWindows {
public:
	static void insert(const std::string& name, const cv::Mat& image);
	static void setX(int x);
	static void setY(int y);
	static void imshow(const std::vector <std::string>& names);
	static void waitKey(int w = 0);
	static void incY();
 private:
	static std::unordered_map <std::string, cv::Mat> m_NamedWindows;
	static int m_X, m_Y, m_XReset, m_YMax;
	const static int m_GapX = 20;
	const static int m_GapY = 40;
};

///////////////////////////////

std::tuple<int, int> findNearestPointOnContours(
	const std::vector <std::vector<cv::Point>>& contours, 
	const cv::Point& p0);

std::tuple <std::tuple <int, int>, std::tuple<int, int>> findNearestPointPairsOnContours(
	const std::vector <std::vector<cv::Point>>& contours, 
	const std::tuple<int, int>& pc0_, 
	const std::tuple<int, int>& pc1_, 
	const int w);

std::vector <cv::Point> closedContourToOpenContour(const std::vector <cv::Point>& closed);

///////////////////////////////

cv::Mat drawWatershedMarkers(
	const cv::Mat& markers, const cv::Mat& image, int th=2, int maxMarker = INT_MAX
);

///////////////////////////////

template <>
struct std::hash<std::tuple<int, int>>
{
	std::size_t operator()(const std::tuple<int, int>& tii) const
	{
		using std::size_t;
		using std::hash;
		using std::string;
		return ((hash<int>()(get<0>(tii)) ^ (hash<int>()(get<1>(tii)) << 1)) >> 1);
    }
};



