
#include <iostream>
#include <vector>
#include <sstream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define USE_GPU 1

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = ::max(::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

static void colorizeFlow(const Mat &u, const Mat &v, Mat &dst)
{
    double uMin, uMax;
    minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = ::abs(uMin); uMax = ::abs(uMax);
    vMin = ::abs(vMin); vMax = ::abs(vMax);
    float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

int main(int argc, char **argv)
{
    VideoCapture cap(0);
    if( !cap.isOpened() )
        return -1;
    // cap.set(CV_CAP_PROP_FRAME_WIDTH, 360);
    // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);


#if USE_GPU
    cout << "CUDA dev count:" << gpu::getCudaEnabledDeviceCount() <<endl;
#endif

    Mat frameL, frameR;
#if USE_GPU
    GpuMat d_frameL, d_frameR;
    GpuMat d_flowx, d_flowy;
    FarnebackOpticalFlow d_calc;
#endif
    Mat flowxy, flowx, flowy, image;

    bool running = true, gpuMode = false;
    int64 t, t0=0, t1=1, tc0, tc1;

    cout << "Use 'm' for CPU/GPU toggling\n";

    namedWindow("flow", 1);

    // for (;;) {
    //   cap >> image;
    //   imshow("flow", image);
    // }

    while (running)
    {
        t = getTickCount();

	cap >> image;
        cvtColor(image, frameR, COLOR_BGR2GRAY);

        if (gpuMode)
        {
            tc0 = getTickCount();

	    if (frameL.data) {
#if USE_GPU
	      d_frameL.upload(frameL);
	      d_frameR.upload(frameR);
	      d_calc(d_frameL, d_frameR, d_flowx, d_flowy);
	      tc1 = getTickCount();
	      d_flowx.download(flowx);
	      d_flowy.download(flowy);

	      cvtColor(frameL, image, COLOR_GRAY2BGR);
	      Mat planes[] = {flowx, flowy};
	      merge(planes, 2, flowxy);
	      drawOptFlowMap(flowxy, image, 16, 1.5, Scalar(0, 255, 0));
	      // colorizeFlow(flowx, flowy, image);
#endif
	    }
        }
        else
        {
            tc0 = getTickCount();
	    if (frameL.data) {
	      calcOpticalFlowFarneback(
				       frameL, frameR, flowxy, 0.5, 3, 15, 3, 5, 1.2, 0);
	      tc1 = getTickCount();
	      
	      Mat planes[] = {flowx, flowy};
	      split(flowxy, planes);
	      flowx = planes[0]; flowy = planes[1];

	      cvtColor(frameL, image, COLOR_GRAY2BGR);
	      drawOptFlowMap(flowxy, image, 16, 1.5, Scalar(0, 255, 0));
	      // colorizeFlow(flowx, flowy, image);
	    }
        }

        stringstream s;
        s << "mode: " << (gpuMode?"GPU":"CPU");
        putText(image, s.str(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        s.str("");
        s << "opt. flow FPS: " << (getTickFrequency()/(tc1-tc0));
        putText(image, s.str(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        s.str("");
        s << "total FPS: " << (getTickFrequency()/(t1-t0));
        putText(image, s.str(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);


        s.str("");
        s << "Yejun & Sejun";
        putText(image, s.str(), Point(5, 145), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

	imshow("flow", image);

        char ch = (char)waitKey(3);
        if (ch == 27)
            running = false;
        else if (ch == 'm' || ch == 'M')
            gpuMode = !gpuMode;

        t0 = t;
        t1 = getTickCount();

	std::swap(frameL, frameR);
    }

    return 0;
}
