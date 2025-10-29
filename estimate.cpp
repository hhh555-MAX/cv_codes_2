#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
//使用BM算法
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

    int blockSize = 9;          // 匹配块大小
    int minDisparity = 0;       // 最小视差值
    int numDisparities = 64;    // 视差范围
    int uniquenessRatio = 15;   // 唯一性阈值
    int speckleWindowSize = 100;// 噪声区域大小
    int speckleRange = 32;      // 噪声允许的视差波动

    // BM算法,设置参数
    Ptr<StereoBM> stereoBM = StereoBM::create(numDisparities, blockSize);
    stereoBM->setMinDisparity(minDisparity);
    stereoBM->setUniquenessRatio(uniquenessRatio);
    stereoBM->setSpeckleWindowSize(speckleWindowSize);
    stereoBM->setSpeckleRange(speckleRange);

    // 计算原始视差图
    Mat dispRaw;
    stereoBM->compute(imgLeft, imgRight, dispRaw);

    // 视差图优化与可视化
    Mat dispNorm, dispFiltered;
    // 归一化
    normalize(dispRaw, dispNorm, 0, 255, NORM_MINMAX, CV_8UC1);
    // 中值滤波
    medianBlur(dispNorm, dispFiltered, 3);

    // 显示结果
    imshow(" left_image", imgLeft);//左图
    imshow(" right_image", imgRight);//右图
    imshow(" result", dispFiltered); //视差图

    //保存视差图
    imwrite("D:/disparity_map.png", dispFiltered);
    cout << "视差图已保存为: D:/disparity_map.png" << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}
