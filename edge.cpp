#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

/// Global variables
Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
string window_name = "Edge Map";

int64 t0 = 0, t1 = 1, tc0, tc1;

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{

  tc0 = getTickCount();
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);

  tc1 = getTickCount();
  stringstream s;

#if 0
  s << "mode: " << (gpuMode ? "GPU" : "CPU") << ", eye: " << (eyeDetection ?
							      "On" : "Off");
  putText (dst, s.str (), Point (5, 25), FONT_HERSHEY_SIMPLEX, 1.,
	   Scalar (255, 0, 255), 2);
#endif

  s.str ("");
  s << "edge detection FPS: " << (getTickFrequency () / (tc1 - tc0));
  putText (dst, s.str (), Point (5, 65), FONT_HERSHEY_SIMPLEX, 1.,
	   Scalar (255, 0, 255), 2);

#if 0
  s.str ("");
  s << "total FPS: " << (getTickFrequency () / (t1 - t0));
  putText (dst, s.str (), Point (5, 105), FONT_HERSHEY_SIMPLEX, 1.,
	   Scalar (255, 0, 255), 2);
#endif

  imshow( window_name, dst );
}


int detectAndDisplay(int )
{
  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Show the image
  CannyThreshold(0, 0);
  // imshow(window_name, src);

  return 0;
}


/** @function main */
int
main (int argc, const char **argv)
{
  CvCapture *
    capture;

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  //-- 2. Read the video stream
  capture = cvCaptureFromCAM (-1);
  if (capture)
    {
      while (true)
	{
	  src = cvQueryFrame (capture);

	  //-- 3. Apply the classifier to the frame
	  if (!src.empty ()) {
	    detectAndDisplay (0);
	  }
	  else  {
	    printf (" --(!) No captured frame -- Break!");
	    break;
	  }

	  int c = waitKey (10);
	  if ((char) c == 'c')   {
	      break;
	    }
	}
    }
  return 0;
}

