#include "utils.h"

using namespace cv;
using namespace std;


/////////////////////////////////////////
// random colors

std::unordered_map<int, cv::Vec3b> Colors::m_Colors;
Colors::Colors(int N) {
	N = max(N, 4);
	for (int n = 0; n < N; n++) {
		m_Colors[n] = Vec3b(rand() % 255, rand() % 255, rand() % 255);
	}
}

cv::Vec3b Colors::at(int n) {
	if (!m_Colors.size())
		Colors();
	return m_Colors[n % m_Colors.size()];
}

cv::Vec3b Colors::at() {
	if (!m_Colors.size())
		Colors();
	return m_Colors[rand() % m_Colors.size()];
}

/////////////////////////////////////////
// named window manager, just for drawing images

std::unordered_map <std::string, cv::Mat> NamedWindows::m_NamedWindows;
int NamedWindows::m_X = 0;
int NamedWindows::m_XReset = 0;
int NamedWindows::m_Y = 0;
int NamedWindows::m_YMax = 0;

void NamedWindows::insert(const std::string& name, const cv::Mat& image) {
	m_NamedWindows[name] = image.clone();
}
void NamedWindows::setX(int x) {
	m_X = x;
	m_XReset = x;
}
void NamedWindows::setY(int y) {
	m_Y = y;
	m_YMax = 0;
}

void NamedWindows::incY() {
	m_Y += m_YMax + m_GapY;
	m_YMax = 0;
}

void NamedWindows::imshow(const std::vector <std::string>& names) {
	for (auto& name : names) {
		if (!m_NamedWindows.count(name))
			continue;
		cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
		cv::moveWindow(name, m_X, m_Y);
		cv::imshow(name, m_NamedWindows[name]);
		if (m_NamedWindows[name].rows > m_YMax)
			m_YMax = m_NamedWindows[name].rows;
		m_X += m_NamedWindows[name].cols + m_GapX;
		if (m_X > 1600) {
			m_X = m_XReset;
			NamedWindows::incY();
		}
	}
}
void NamedWindows::waitKey(int w) {
	cv::waitKey(w);
}


/////////////////////////////////////////
// given a single source point, finds the nearest
// point on any contour in a vector of contours

std::tuple<int, int> findNearestPointOnContours(
	const std::vector <std::vector<cv::Point>>& contours, 
	const cv::Point& p0)
{
	tuple <int, int> pP = {-1, -1};
	int d2_min = INT_MAX;
	for (int c = 0; c < contours.size(); c++) {
		for (int i = 0; i < contours.at(c).size(); i++) {
			Point p1 = contours.at(c).at(i);
			int d2 = ((p1.x - p0.x) * (p1.x - p0.x)) + ((p1.y - p0.y) * (p1.y - p0.y));
			if (d2 < d2_min)
			{
				d2_min = d2;
				pP = make_tuple(c, i);
			}
		}
	}
	return(pP);
}

/////////////////////////////////////////
// given two initial points on contours,
// searchs near those points for their closest
// pair

std::tuple <std::tuple <int, int>, std::tuple<int, int>> findNearestPointPairsOnContours(
	const std::vector <std::vector<cv::Point>>& contours, 
	const std::tuple<int, int>& pc0_, 
	const std::tuple<int, int>& pc1_, 
	const int w)
{
	tuple <int, int> pc0o = { -1,-1 };
	tuple <int, int> pc1o = { -1,-1 };
	auto& c0_ = get<0>(pc0_);
	auto& p0_ = get<1>(pc0_);
	auto& c1_ = get<0>(pc1_);
	auto& p1_ = get<1>(pc1_);

	int d2_min = INT_MAX;
	for (int d0 = -w; d0 <= +w; d0++) {

		int p0p = p0_ - d0;
		if (p0p < 0)
			p0p += (int) contours.at(c0_).size();
		if (p0p >= contours.at(c0_).size())
			p0p -= (int)contours.at(c0_).size();

		for (int d1 = -w; d1 <= +w; d1++) {
			int p1p = p1_ - d1;
			if (p1p < 0)
				p1p += (int)contours.at(c1_).size();
			if (p1p >= contours.at(c1_).size())
				p1p -= (int)contours.at(c1_).size();

			Point p0t = contours.at(c0_).at(p0p);
			Point p1 = contours.at(c1_).at(p1p);
			int d2 = ((p1.x - p0t.x) * (p1.x - p0t.x)) + ((p1.y - p0t.y) * (p1.y - p0t.y));
			if (d2 < d2_min) {
				d2_min = d2;
				pc0o = make_tuple(c0_, p0p);
				pc1o = make_tuple(c1_, p1p);
			}
		}
	}
	return(make_tuple(pc0o, pc1o));
}


/////////////////////////////////////////
// turns a 2D contour into a 1D contour
// the contour must be found from an image containing
// a skeletonized (line-8 style) polygon

std::vector <cv::Point> closedContourToOpenContour(const std::vector <cv::Point>& closed)
{
	vector <Point> open;
	open.reserve((closed.size() / 2) + 1);
	bool started = false;

	for (int p = 0; p < closed.size(); p++)	{
		int pp, pm;
		if (p == 0)	{
			pm = (int)closed.size() - 1;
			pp = 1;
		}
		else if (p == closed.size() - 1) {
			pm = (int)closed.size() - 2;
			pp = 0;
		}
		else {
			pm = p - 1;
			pp = p + 1;
		}

		if (!started && (closed.at(pm) == closed.at(pp))) {
			open.push_back(closed.at(p));
			started = true;
		}
		else if (started) {
			open.push_back(closed.at(p));
			if (closed.at(pm) == closed.at(pp))
				return(open);
		}
	}
	return(closed);
}

/////////////////////////////////////////
// draws watershed markers on an image. kinda similar to extract objects

cv::Mat drawWatershedMarkers(const cv::Mat& markers, const cv::Mat& image, int th, int maxMarker) {

	unordered_set <int> objectIndicees;
	for (int r = 0; r < markers.rows; r++) {
		for (int c = 0; c < markers.cols; c++) {
			int objectIndex = markers.at<int>(r, c);

			if ((objectIndex == 0) ||
				(objectIndex >= (int) maxMarker) ||
				(objectIndex == -1))
				continue;
			objectIndicees.insert(objectIndex);
		}
	}

	Mat imageDraw = image.clone();
	int numObjects = *std::max_element(objectIndicees.begin(), objectIndicees.end());
	for (int i=1; i<=numObjects; i++) {
		Mat mask = (markers == i);
		vector <vector <Point>> contours;
		findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		drawContours(imageDraw, contours, -1, Colors::at(i), 2);			
	}

	return imageDraw;
}

/////////////////////////////////////////
// extract objects 

void extractObjects(const cv::Mat& image, const cv::Mat& markers, std::vector <cv::Mat> objects) {
	int objectNum = 1;
	do {
		Mat mask = (markers == objectNum++);
		if (countNonZero(mask) == 0)
			break;
		Mat object;
		image.copyTo(object, mask);
		objects.push_back(object);
	} while(true);
}


