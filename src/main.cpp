#include "../watershed_r2/utils.h"
#include "../watershed_r2/watershed_r2.h"

#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main()
{
    // read the image
    Mat coins = imread("../../Watershed_r2/water_coins.jpg");

    // convert the image into objects
    vector <Mat> objects;
    findObjects(coins, objects);

    // display the objects (and the temporary watershed images)
    NamedWindows::setX(200);
    NamedWindows::setY(100);
    NamedWindows::imshow({"watershed", "borders", "fixed borders", "fixed watershed"});

    vector <string> objectNames = {};
	for (int i=0; i<=objects.size(); i++)
		objectNames.push_back("object " + to_string(i));
    NamedWindows::setX(200);
    NamedWindows::incY();
    NamedWindows::imshow(objectNames);

    NamedWindows::waitKey();
}
