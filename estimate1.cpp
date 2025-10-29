#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
//使用半全局匹配方法
int main()
{
    String img_left = "D:/work/tsukuba_l.png";
    String img_right = "D:/work/tsukuba_r.png";
    
    //读取左右图片
    Mat imgLeft = imread(img_left, IMREAD_GRAYSCALE);  // 直接得到单通道
    Mat imgRight = imread(img_right, IMREAD_GRAYSCALE);

    // 检查左右图尺寸是否一致
    if (imgLeft.size() != imgRight.size()) 
    {
        cout << "错误：左右图片尺寸不一致，无法进行视差估计！" << endl;
        return -1;
    }

    // SGBM参数设置
    int minDisparity = 0;               // 最小视差值
    int numDisparities = 64;           // 视差范围
    int blockSize = 3;                  // 匹配块大小
    
    // 预处理参数
    int P1 = 8 * 3 * blockSize * blockSize;  // 相邻像素视差变化为1时的惩罚项
    int P2 = 32 * 3 * blockSize * blockSize; // 相邻像素视差变化大于1时的惩罚项（需大于P1）
    
    // 后处理参数
    int disp12MaxDiff = 1;              // 左右视差一致性检查的最大允许差异
    int uniquenessRatio = 10;           // 唯一性校验阈值
    int speckleWindowSize = 100;        // 平滑噪声区域的窗口大小
    int speckleRange = 16;              // 噪声区域内视差的最大允许变化

    // 使用SGBM，设置参数
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        disp12MaxDiff,
        uniquenessRatio,
        speckleWindowSize,
        speckleRange
    );

    Mat dispRaw;
    sgbm->compute(imgLeft, imgRight, dispRaw);
    Mat dispNorm;
    
    normalize(dispRaw, dispNorm, 0, 255, NORM_MINMAX, CV_8UC1);

    imshow(" left_image", imgLeft);//左图
    imshow(" right_image", imgRight);//右图
    imshow("result", dispNorm);

    imwrite("D:/sgbm_gray_disparity.png", dispNorm);
    cout << "视差图已保存！" << endl;

    waitKey(0);
    destroyAllWindows();
    return 0;
}