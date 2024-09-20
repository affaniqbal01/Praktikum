#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <chrono>
 
using namespace cv;
using namespace std;

constexpr int repetitions = 10000;
 
int main( int argc, char** argv )
{
    Mat src = imread( "/home/gles/Praktikum/WarpAffine/cat-1.jpg" );
    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
 
    Point2f srcTri[3];
    srcTri[0] = Point2f( 0.f, 0.f );
    srcTri[1] = Point2f( src.cols - 1.f, 0.f );
    srcTri[2] = Point2f( 0.f, src.rows - 1.f );
 
    Point2f dstTri[3];
    dstTri[0] = Point2f( 0.f, src.rows*0.33f );
    dstTri[1] = Point2f( src.cols*0.95f, src.rows*0.25f );
    dstTri[2] = Point2f( src.cols*0.5f, src.rows*0.7f );
 
    Mat warp_mat = getAffineTransform( srcTri, dstTri );
 
    Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < repetitions; ++i)
    {
        warpAffine(src, warp_dst, warp_mat, warp_dst.size() );
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto diff = stop - start;
    std::cout << "process took by CPU: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
    << " milliseconds\n";


    ///// GPU

    cv::cuda::GpuMat gpuDst, gpuSrc;
    gpuSrc.upload(src);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < repetitions; ++i)
    {
        cv::cuda::warpAffine(gpuSrc, gpuDst, warp_mat, warp_dst.size());
    }
    stop = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    gpuDst.download(result);

    diff = stop - start;
    std::cout << "process took by GPU: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
    << " milliseconds\n";
 
    // Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
    // double angle = -50.0;
    // double scale = 0.6;
 
    // Mat rot_mat = getRotationMatrix2D( center, angle, scale );
 
    // Mat warp_rotate_dst;
    // warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );
 
    imshow( "Source image", result );
    // imshow( "Warp", warp_dst );
    // imshow( "Warp + Rotate", warp_rotate_dst );
 
    waitKey();
 
    return 0;
}