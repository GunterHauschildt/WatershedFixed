#include "watershed_r2.h"
#include "utils.h"
#include "opencv2/highgui.hpp"
#include <string>

using namespace std;
using namespace cv;

#define DEBUG_WATERSHED_DRAW

/////////////////////////////////////////
// segment an object known where the a mask containg
// the objects are touching. Recreates a border
// similar to watershed's border (except uses
// a LINE_4 type instead of LINE_8) and then
// just calls findContours
// TODO: could return the contours instead of the
// watershed markers

void static segmentByBorders(
	const cv::Mat& mask_, 
	const std::vector<std::tuple<cv::Point, cv::Point>>& borders, 
	cv::Mat& markers,
	int th=1) {

	auto extendLineSegment = [th](
		std::tuple <cv::Point, cv::Point> ps) {

			auto& A = get<0>(ps);
			auto& B = get<1>(ps);
			Point C, D;
			double lenAB = sqrt(pow(A.x - B.x, 2.0) + pow(A.y - B.y, 2.0));
			double dx = (B.x - A.x) / lenAB * th;
			double dy = (B.y - A.y) / lenAB * th;
			C.x = cvRound(A.x - dx);
			C.y = cvRound(A.y - dy);
			D.x = cvRound(B.x + dx);
			D.y = cvRound(B.y + dy);
			return make_tuple(C, D);
		};

	// recreate the border.
	// LINE_4 is important to ensure no pixels touch on the diagonal.
	auto mask = mask_.clone();
	for (auto& border : borders) {
		auto extendedBorder = extendLineSegment(border);
		line(mask, get<0>(extendedBorder), get<1>(extendedBorder), Scalar::all(0), th, cv::LINE_4);
	}

	vector <vector <Point>> contours;
	rectangle(mask, Point(0,0), Point(mask.cols-1, mask.rows-1), Scalar::all(0), 1);

	// findContours needs a LINE_8 seperate and the original watershed mask gives LINE_4
	morphologyEx(mask, mask, cv::MORPH_ERODE, getStructuringElement(cv::MORPH_RECT, Size(3, 3)));
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	// it'd be smart to return the contours here as thats what we're likely ultimately
	// interested in. in this demo though we care about the markers.
	// we could/ should also regrow the markers
	markers.setTo(Scalar::all(0));
	for (int c = 0; c < contours.size(); c++) {
		drawContours(markers, contours, c, Scalar::all(c + 1), cv::FILLED);
	}
}

/////////////////////////////////////////
// Using a 3x3 kernel around a given known watershed marker,
// finds the border and its neighbour markers
// returns as a map of neighbour pairs and their border

void static findBordersAndNeighbours(
	const cv::Mat& markers, 
	int r, int c,
	std::unordered_map <std::tuple <int, int>, WatershedObjectBorder>& objectBorders,
	int maxObjectIndex=INT_MAX
	)
{
	// find the borders between the objects. we use this if calling fixWatershed()
	// basically we're walking around a 3x3 kernel and finding if there's a neighbouring
	// pixel. is so its a neighbour. we also find the center of that neighbour
	// in a bounding box sense so we know how to find which side of the (upcoming)
	// dividing line it is on

	int neighborIndex_0 = -1;
	int neighborIndex_1 = -1;

	tuple <int, int> neighbors = make_tuple(-1, -1);
	for (int rp_ = r - 1; rp_ <= r + 1; rp_++)
	{
		for (int cp_ = c - 1; cp_ <= c + 1; cp_++)
		{
			// not likely to happen (as seperating won't be at the edge) but just in case

			int rp = std::max(rp_, 0);
			rp = min(rp, markers.rows - 1);
			int cp = std::max(cp_, 0);
			cp = min(cp, markers.cols - 1);

			int neighborIndex = markers.at<int>(rp, cp);

			if ((neighborIndex == -1) || (neighborIndex == 0) || (neighborIndex >= maxObjectIndex))
				continue;

			if (neighborIndex_0 == -1)
				neighborIndex_0 = neighborIndex;

			else if ((neighborIndex_0 != -1) && (neighborIndex != neighborIndex_0))
			{
				neighborIndex_1 = neighborIndex;
				int min = std::min(neighborIndex_0, neighborIndex_1);
				int max = std::max(neighborIndex_0, neighborIndex_1);

				neighbors = make_tuple(min, max);
				break;
			}
		}
		if ((get<0>(neighbors) != -1) && (get<1>(neighbors) != -1))
			break;
	}

	if ((get<0>(neighbors) != -1) && (get<1>(neighbors) != -1))
	{
		if (!objectBorders.count(neighbors))
			objectBorders[neighbors] = WatershedObjectBorder(markers.size());
		objectBorders[neighbors].mask.at<uchar>(r, c) = (uchar)255;

		if (c < objectBorders[neighbors].tl.x)
			objectBorders[neighbors].tl.x = c;

		if (c > objectBorders[neighbors].br.x)
			objectBorders[neighbors].br.x = c;

		if (r < objectBorders[neighbors].tl.y)
			objectBorders[neighbors].tl.y = r;

		if (r > objectBorders[neighbors].br.y)
			objectBorders[neighbors].br.y = r;
	}

}

/////////////////////////////////////////
// The main algorithm:
// After watershed and a normal-ish watershed reconstruction
// with the addition of finding the borders and neighbours
// - turns those border Mats into 2D contours
// - turns those 2D contours into 1D contours
// - finds the best find pairs for those 1D contours, assuming
// that the best fit is defined by 'closest'
// - resegments based on those closet paairs 

void fixWatershedSeperatedObjects(
	const cv::Mat& image,
	const cv::Mat& mask,
	const std::unordered_map <std::tuple<int, int>, WatershedObjectBorder>& objectBorders,
	cv::Mat& markers)
{
#ifdef DEBUG_WATERSHED_DRAW
	Mat drawWatershedBorder = image.clone();
	Mat drawFixedBorder = image.clone();
#endif

	vector <vector <Point>> allContours;
	findContours(mask, allContours, RETR_CCOMP, CHAIN_APPROX_NONE);
	vector <tuple <Point, Point>> fixedBorders;

	for (auto& objectBorder_ : objectBorders)
	{
		auto& objects = objectBorder_.first;
		auto& objectBorder = objectBorder_.second;
		vector <vector <Point>> objectBorder_contours;

		// look in the just the area defined by the border
		Point fcROI_tl = objectBorder.tl - Point(1, 1);
		Point fcROI_br = objectBorder.br + Point(2, 2);
		findContours(
			objectBorder.mask(Rect(fcROI_tl, fcROI_br)).clone(), 
			objectBorder_contours, 
			cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, fcROI_tl
		);

#ifdef DEBUG_WATERSHED_DRAW
		drawContours(drawWatershedBorder, objectBorder_contours, 0, Colors::at(), 2);
#endif
		// a contour is always closed by defn in open CV.
		// ie, for a known 1-pixel curve, it will be the same points repeated
		// so get just the curve
		vector <Point> border = closedContourToOpenContour(objectBorder_contours.at(0));

		// find the nearest points to the start and end of the border
		auto p0 = border.front();
		auto p1 = border.back();
		auto pp0p = findNearestPointOnContours(allContours, p0);
		auto pp1p = findNearestPointOnContours(allContours, p1);

		// find the nearest pairs to the nearest points on the contour
		// that becomes the pair to separate
		// 'w' is the search window
		int w = max(cvRound(double(image.cols) * .05), 5);
		auto ppps = findNearestPointPairsOnContours(allContours, pp0p, pp1p, w);
		auto& pp0pp = get<0>(ppps);
		auto& pp1pp = get<1>(ppps);

		Point p0pp = allContours.at(get<0>(pp0pp)).at(get<1>(pp0pp));
		Point p1pp = allContours.at(get<0>(pp1pp)).at(get<1>(pp1pp));

		// just in case
		if (p0pp == p1pp) {
			continue;
		}

		// we're good with fixing this border
		fixedBorders.push_back(make_tuple(p0pp, p1pp));

#ifdef DEBUG_WATERSHED_DRAW
		line(drawFixedBorder, p0pp, p1pp, Colors::at(), 2);
#endif
	}

#ifdef DEBUG_WATERSHED_DRAW
	NamedWindows::insert("borders", drawWatershedBorder);
	NamedWindows::insert("fixed borders", drawFixedBorder);
#endif

	// and do the final resegmentation after we've fixed all the borders
	segmentByBorders(mask, fixedBorders, markers);
}


/////////////////////////////////////////
// extract objects 

static void extractObjects(const cv::Mat& image, const cv::Mat& markers, std::vector <cv::Mat>& objects) {

	int objectNum = 1;
	do {
		Mat mask = (markers == objectNum);
		if (countNonZero(mask) == 0)
			break;
		morphologyEx(mask, mask, cv::MORPH_ERODE, getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3)));
		Rect r = boundingRect(mask);
		Mat object;
		image(r).copyTo(object, mask(r));
		objects.push_back(object);
#ifdef DEBUG_WATERSHED_DRAW
		// the objects are a little too small for a demo
		Mat object_p;
		resize(object, object_p, Size(), 3.0, 3.0);
		NamedWindows::insert("object " + to_string(objectNum), object_p);
#endif
		objectNum++;
	} while (1);
}


/////////////////////
// the main external call.
// real use would probably return the individual objects here.
// the first half is similar to OpenCV's
// https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html
// then we fix the borders with the algorithm described here

void findObjects(const cv::Mat& image, std::vector <cv::Mat>& objects) {
	
	// hard code a magic number (for this demo only)
	double distanceTransformThreshold = 0.5;

	// find the initial guess by thresholding with otsu
	Mat mask;
	cvtColor(image, mask, cv::COLOR_BGR2GRAY);
	cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

	// distance transform & threshold to find foreground
	Mat fDXform;
	distanceTransform(mask, fDXform, cv::DIST_L1, 5);
	double max;
	minMaxLoc(fDXform, NULL, &max, NULL, NULL);
	threshold(fDXform, fDXform, distanceTransformThreshold * max, 255, cv::THRESH_BINARY);
	fDXform.convertTo(fDXform, CV_8UC1);
	vector <vector <Point>> DXformContours;
	findContours(fDXform, DXformContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// INT_MAX = the background
	Mat watershedMarkers32S = INT_MAX * Mat::ones(mask.size(), CV_32S);
	morphologyEx(mask, mask, MORPH_DILATE, 
		getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), Point(-1, -1), 2, 
		BORDER_CONSTANT, Scalar::all(0)
	);

	// 0 = the unknown area (so the dilated mask)
	watershedMarkers32S.setTo(0, mask);

	// N = the known area (the distance xformed contours)
	for (int i = 0; i < DXformContours.size(); i++)
		drawContours(watershedMarkers32S, DXformContours, i, Scalar::all(i + 1), cv::FILLED);

	Mat watershedMarkers8U;
	watershedMarkers32S.convertTo(watershedMarkers8U, CV_8UC3);	

	// do watershed'ing
	watershed(image, watershedMarkers32S);

#ifdef DEBUG_WATERSHED_DRAW
	Mat drawWatershedObjects = drawWatershedMarkers(watershedMarkers32S, image);
	NamedWindows::insert("watershed", drawWatershedObjects);
#endif

	// outer loop: 
	// is the normal method of extracting masks from the watershed markers
	unordered_map <tuple <int, int>, WatershedObjectBorder> objectBorders;
	Mat objectsMask = Mat::zeros(watershedMarkers32S.size(), CV_8UC1);
	for (int r = 0; r < watershedMarkers32S.rows; r++) {
		for (int c = 0; c < watershedMarkers32S.cols; c++) {

			int waterShedMarkerObjectIndex = watershedMarkers32S.at<int>(r, c);

			if ((waterShedMarkerObjectIndex == 0) ||
				(waterShedMarkerObjectIndex > (int) DXformContours.size()))
				continue;

			objectsMask.at<uchar>(r, c) = (uchar) 255;

			if (waterShedMarkerObjectIndex == -1) {
				// inner loop
				// find the borders and their neigbours between the objects
				findBordersAndNeighbours(
					watershedMarkers32S, r, c, objectBorders
				);
				continue;
			}
		}
	}

	// now finally fix the segmentation borders
	fixWatershedSeperatedObjects(image, objectsMask, objectBorders, watershedMarkers32S);

#ifdef DEBUG_WATERSHED_DRAW
	drawWatershedObjects = drawWatershedMarkers(watershedMarkers32S, image);
	NamedWindows::insert("fixed watershed", drawWatershedObjects);
#endif

	// extract the objects
	extractObjects(image, watershedMarkers32S, objects);
}
