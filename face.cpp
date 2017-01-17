#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <stdio.h>

#define USE_CAMERA 1
#define USE_DISP 1

using namespace std;
using namespace cv;
using namespace cv::gpu;

/** Function Headers */
void
detectAndDisplay (Mat frame);

/** Global variables */
String
  face_cascade_name = "haarcascade_frontalface_alt.xml";
String
  eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier  face_cascade;
CascadeClassifier  eyes_cascade;

gpu::CascadeClassifier_GPU face_cascade_gpu;
gpu::CascadeClassifier_GPU eyes_cascade_gpu;

string
  window_name = "Capture - Face detection";
RNG
rng (12345);

int64
  t,
  t0 = 0, t1 = 1, tc0, tc1;
int
  gpuMode = 1;
int
  eyeDetection = 1;

 /** @function main */
int
main (int argc, const char **argv)
{
  CvCapture *    capture;
  Mat    frame;

  //-- 1. Load the cascades
  if (!face_cascade.load (face_cascade_name))
    {
      printf ("--(!)Error loading\n");
      return -1;
    };
  if (!eyes_cascade.load (eyes_cascade_name))
    {
      printf ("--(!)Error loading\n");
      return -1;
    };


  if (!face_cascade_gpu.load (face_cascade_name))
    {
      printf ("--(!)Error loading\n");
      return -1;
    };

  if (!eyes_cascade_gpu.load (eyes_cascade_name))
    {
      printf ("--(!)Error loading\n");
      return -1;
    };

  //-- 2. Read the video stream
#if USE_CAMERA==1
  capture = cvCaptureFromCAM (-1);
#else
  capture = cvCaptureFromFile(argv[1]);
#endif
  if (capture)
    {
      while (true)
	{
	  frame = cvQueryFrame (capture);

	  
	  //-- 3. Apply the classifier to the frame
	  if (!frame.empty ())
	    {
	      detectAndDisplay (frame);
	    }
	  else
	    {
	      // printf (" --(!) No captured frame -- Break!");
	      break;
	    }

	  int
	    c = waitKey (10);
	  if ((char) c == 'c')
	    {
	      break;
	    }
	}
    }
  return 0;
}

/** @function detectAndDisplay */
void
detectAndDisplay (Mat frame)
{
  std::vector < Rect > faces;
  Mat
    frame_gray;

  t0 = getTickCount ();
  cvtColor (frame, frame_gray, CV_BGR2GRAY);
  equalizeHist (frame_gray, frame_gray);

  //-- Detect faces
  if (gpuMode)
    {
      tc0 = getTickCount ();

      GpuMat
      frame_gray_gpu (frame_gray);
      GpuMat
	faces_gpu;
      int
	nface =
	face_cascade_gpu.detectMultiScale (frame_gray_gpu, faces_gpu, 1.1, 2,
					   Size (30, 30));

      Mat
	obj_host;
      faces_gpu.colRange (0, nface).download (obj_host);

      Rect *
	facesPtr = obj_host.ptr < Rect > ();
      for (int i = 0; i < nface; i++)
	faces.push_back (facesPtr[i]);

      tc1 = getTickCount ();
    }
  else
    {
      tc0 = getTickCount ();
      face_cascade.detectMultiScale (frame_gray, faces, 1.1, 2,
				     0 | CV_HAAR_SCALE_IMAGE, Size (30, 30));
      tc1 = getTickCount ();
    }

  for (size_t i = 0; i < faces.size (); i++)
    {
      Point
      center (faces[i].x + faces[i].width * 0.5,
	      faces[i].y + faces[i].height * 0.5);
      ellipse (frame, center,
	       Size (faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360,
	       Scalar (255, 0, 255), 4, 8, 0);

      if (eyeDetection)
	{
	  Mat
	    faceROI = frame_gray (faces[i]);
	  std::vector < Rect > eyes;

	  if (gpuMode)
	    {
	      GpuMat
	      faceROI_gpu (faceROI);
	      GpuMat
		eyes_gpu;
	      int
		neyes =
		eyes_cascade_gpu.detectMultiScale (faceROI_gpu, eyes_gpu, 1.1,
						   2, Size (30, 30));

	      Mat
		obj_host;
	      eyes_gpu.colRange (0, neyes).download (obj_host);

	      Rect *
		eyesPtr = obj_host.ptr < Rect > ();
	      for (int i = 0; i < neyes; i++)
		eyes.push_back (eyesPtr[i]);

	    }
	  else
	    {
	      //-- In each face, detect eyes
	      eyes_cascade.detectMultiScale (faceROI, eyes, 1.1, 2,
					     0 | CV_HAAR_SCALE_IMAGE,
					     Size (30, 30));
	    }

	  for (size_t j = 0; j < eyes.size (); j++)
	    {
	      Point
	      center (faces[i].x + eyes[j].x + eyes[j].width * 0.5,
		      faces[i].y + eyes[j].y + eyes[j].height * 0.5);
	      int
		radius = cvRound ((eyes[j].width + eyes[j].height) * 0.25);
	      circle (frame, center, radius, Scalar (255, 0, 0), 4, 8, 0);
	    }
	}
    }

  t1 = getTickCount ();

  stringstream
    s;

  s << "mode: " << (gpuMode ? "GPU" : "CPU") << ", eye: " << (eyeDetection ?
							      "On" : "Off");
  putText (frame, s.str (), Point (5, 25), FONT_HERSHEY_SIMPLEX, 1.,
	   Scalar (255, 0, 255), 2);

  s.str ("");
  s << "face detection FPS: " << (getTickFrequency () / (tc1 - tc0));
  putText (frame, s.str (), Point (5, 65), FONT_HERSHEY_SIMPLEX, 1.,
	   Scalar (255, 0, 255), 2);

  s.str ("");
  float fps = (getTickFrequency () / (t1 - t0));
  s << "total FPS: " << fps;
  putText (frame, s.str (), Point (5, 105), FONT_HERSHEY_SIMPLEX, 1.,
	   Scalar (255, 0, 255), 2);

  cout << fps << " " << (float)(tc1 - tc0)/1000000 << " " << (float)(t1 - t0)/1000000 << endl;

#if USE_DISP==1
  //-- Show what you got
  imshow (window_name, frame);
#endif
  char
    ch = (char) waitKey (3);
  if (ch == 'm' || ch == 'M')
    gpuMode = !gpuMode;
  else if (ch == 'e' || ch == 'E')
    eyeDetection = !eyeDetection;
}
