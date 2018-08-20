// 检测墙体与柱子的类
#ifndef PLANE_AND_LINE_DETECT
#define PLANE_AND_LINE_DETECT

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <Eigen/Geometry>  //一些矩阵的操作
#include <vector>
#include <math.h>
#include <string>
#include <pcl/visualization/cloud_viewer.h>
#include <ros/assert.h>
#include <Eigen/Eigenvalues>
#include "lidar_localization/PointsClassification.h"

using namespace pcl;
using namespace std;
using namespace Eigen;

#define WIDTH 2016  //
#define HEIGHT 16   //the lines of the lidar
#define DTRED 0.05   // distance threshold
#define NTRED 0.8   // n . n threshold it must smaller than 1

typedef PointXYZI PointI;

struct Bound
{
    int leftW = -1; //left WIDTH boundary
    int rightW = -1;
    int upH = -1; // up pitch Boundary;
    int bottomH =-1;
};

class PlaneAndLineDetect{
public:
    PlaneAndLineDetect(){
        inputPCLptr.reset(new PointCloud<PointI>);
        planeAndLinePCLptr.reset(new PointCloud<PointI>);
        normalPCLptr.reset(new PointCloud<PointNormal>);
        planeAndLine_pub = nh.advertise<sensor_msgs::PointCloud2>("/plane_and_line",1);
        lidarpoints_sub = nh.subscribe("/lidar_points",1,&PlaneAndLineDetect::startDetect,this);
    }
    void startDetect(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg){
        frameID = (*point_cloud_msg).header.frame_id;
        pcl::fromROSMsg(*point_cloud_msg,*inputPCLptr);
        planeAndLinePCLptr->clear();
        normalPCLptr->clear();
        for(int i=0;i<WIDTH;i++){
            for(int j=0;j<HEIGHT;j++){
                depthMat[i][j] = 0;
                indexMat[i][j] =-1;
                vFlagMat[i][j] = false;
                meanMat[i][j] = MatrixXd::Zero(3,1);
                varianceMat[i][j] = Matrix3d::Zero();
                clusterMat[i][j] = 0;
            }
        }
        setDepthAndIndexMap();
        setIntervalMap();
        setIntegralMap();
        setClusterMap();
        classsifyCluster();
        // computeNormal();
    }
    void setDepthAndIndexMap(){
        int chongdie = 0;
        int errorpoint = 0;
        int nannum = 0;
        // one quadurant has WIDTH/4 offset and y = 0 belong to the first
        int row = 0,col = 0;
        int offset = 0;
        double vAngle = 0;
        for(int i = 0;i < inputPCLptr->size() ; ++i){
            double x = inputPCLptr->points[i].x;
            double y =inputPCLptr->points[i].y;
            double z =inputPCLptr->points[i].z;

            ROS_ASSERT_MSG(!(x == 0 && y ==0),"x,y is all 0");
            if( !(x==x) || !(y==y) ||!(z==z)){  //NAN judge
                nannum++;
                continue;
            }
            vAngle = atan2(y , x);
            if(vAngle <0)
                vAngle += 2.0*M_PI;
            row = round((vAngle* 2016.0/(M_PI*2.0)) );
            col = (round(atan2(z,sqrt(x*x+y*y))*180.0/M_PI) +15) / 2;
            if(row<0 || row >= WIDTH || col <0 || col >= HEIGHT){
                errorpoint++;
                continue;
            }
            if(depthMat[row][col] != 0 || indexMat[row][col] != -1){
                chongdie++;
            }else{
                depthMat[row][col] = sqrt(x*x+y*y+z*z);
                indexMat[row][col] = i;
            }
        }
        // cout << "chongdie = " << chongdie << endl;
        PointI middle,now,left,right;
        const int thresholdN = 4; // the shortest
        for(int w = 0;w < WIDTH;w++){
            int verticalNum = 1;
            int wPush = w;
            int wL = w==0 ? WIDTH-1:w-1;
            int wR = (w==WIDTH-1) ? 0:w+1;
            for(int h = 0;h < HEIGHT- thresholdN; h++){
                vector<int> index_vector;
                vector<int> index_vector1;
                verticalNum = 1;
                if(vFlagMat[w][h] || indexMat[w][h] == -1)
                    continue;
                now = inputPCLptr->points[indexMat[w][h]];
                index_vector.push_back(h);
                index_vector1.push_back(w);

                double leftD = 10000,middleD=10000,rightD=10000;
                for(int n = h+1; n < HEIGHT;n++){
                    if(indexMat[wL][n]!=-1){
                        left = inputPCLptr->points[indexMat[wL][n]];
                        leftD = fabs(sqrt(now.x * now.x + now.y * now.y) - sqrt(left.x * left.x +left.y * left.y));
                    }
                    if(indexMat[w][n]!=-1){
                        middle = inputPCLptr->points[indexMat[w][n]];
                        middleD = fabs(sqrt(now.x * now.x + now.y * now.y) - sqrt(middle.x * middle.x +middle.y * middle.y));
                    }
                    if(indexMat[wR][n]!=-1){
                        right = inputPCLptr->points[indexMat[wR][n]];
                        rightD = fabs(sqrt(now.x * now.x + now.y * now.y) - sqrt(right.x * right.x +right.y * right.y));
                    }
                    double distance = middleD;
                    if(leftD < middleD){
                        wPush = wL;
                        distance = leftD;
                    }else if(rightD < distance){
                        wPush = wR;
                        distance = rightD;
                    }
                    if( distance<0.1 ){
                        verticalNum++;
                        index_vector.push_back(n);
                        index_vector1.push_back(wPush);
                    }
                }
                if(verticalNum >= thresholdN){
                    for(int push = 0;push < index_vector.size(); ++push){
                        vFlagMat[index_vector1[push]][index_vector[push]] = true;
                    }
                }else{
                    1;
                    // depthMat[w][h] = 0;
                    // indexMat[w][h] = -1;
                }
            }
        }
    }
    void setIntervalMap(){
        const double depthThreshold = 0.1; // may change it later
        for(int w = 0;w < WIDTH-1;w++){
            for(int h = 0;h < HEIGHT-1;h++){
                if(vFlagMat[w][h]){
                    int l = w,r = w,u = h,b = h;//l left, r right ,u up,b bottom
                    // while(l != 0 && (indexMat[l-1][h] ==-1 || fabs(depthMat[l-1][h] - depthMat[w][h]) < depthThreshold)){
                    while(l != 0 && (fabs(depthMat[l-1][h] - depthMat[w][h]) < depthThreshold)){
                        --l;
                    }
                    // while(r != WIDTH-1 && (indexMat[r+1][h] ==-1 || fabs(depthMat[r+1][h] - depthMat[w][h]) < depthThreshold)){
                    while(r != WIDTH-1 && (fabs(depthMat[r+1][h] - depthMat[w][h]) < depthThreshold)){
                        ++r;
                    }
                    // while(b != 0 && (indexMat[w][b-1] ==-1 || fabs(depthMat[w][b-1] - depthMat[w][h]) < depthThreshold)){
                    while(b != 0 && (fabs(depthMat[w][b-1] - depthMat[w][h]) < depthThreshold)){
                        --b;
                    }
                    // while(u != HEIGHT-1 && (indexMat[w][u+1] ==-1 || fabs(depthMat[w][u+1] - depthMat[w][h]) < depthThreshold)){
                    while(u != HEIGHT-1 && (fabs(depthMat[w][u+1] - depthMat[w][h]) < depthThreshold)){
                        ++u;
                    }
                    boundMat[w][h].leftW = l;
                    boundMat[w][h].rightW = r;
                    boundMat[w][h].upH = u;
                    boundMat[w][h].bottomH = b;
                    // if(w >1542 && w < 1580){
                        // cout << w << ',' << h << endl;
                        // cout << l << ' ' << r << ' ' << u << ' ' << b << endl;
                        // cout << inputPCLptr->points[indexMat[w][h]] << endl;
                    // }
                }
            }
        }
    }
    // ji fen tu
    void setIntegralMap(){
        if(indexMat[0][0] != -1 && vFlagMat[0][0]){
            meanMat[0][0](0) = inputPCLptr->points[indexMat[0][0]].x;
            meanMat[0][0](1) = inputPCLptr->points[indexMat[0][0]].y;
            meanMat[0][0](2) = inputPCLptr->points[indexMat[0][0]].z;
            // 一个点的时候方差是0;
            varianceMat[0][0] = meanMat[0][0] * meanMat[0][0].transpose();
        }
        Vector3d tempV3d = Vector3d::Zero();
        // initial first row and col
        for(int w = 1;w < WIDTH;w++){
            meanMat[w][0] = meanMat[w-1][0];
            if(indexMat[w][0] != -1 && vFlagMat[w][0]){
                tempV3d(0) = inputPCLptr->points[indexMat[w][0]].x;
                tempV3d(1) = inputPCLptr->points[indexMat[w][0]].y;
                tempV3d(2) = inputPCLptr->points[indexMat[w][0]].z;
                meanMat[w][0] += tempV3d;
                varianceMat[w][0] +=  tempV3d * tempV3d.transpose();
            }
        }
        for(int h=1;h < HEIGHT;h++){
            meanMat[0][h] = meanMat[0][h-1];
            if(indexMat[0][h] != -1 && vFlagMat[0][h]){
                tempV3d(0) = inputPCLptr->points[indexMat[0][h]].x ;
                tempV3d(1) = inputPCLptr->points[indexMat[0][h]].y ;
                tempV3d(2) = inputPCLptr->points[indexMat[0][h]].z ;
                meanMat[0][h] += tempV3d;
                varianceMat[0][h] += tempV3d* tempV3d.transpose();
            }
        }
        for(int w = 1;w < WIDTH; w++){
            for(int h = 1; h < HEIGHT; h++){
                meanMat[w][h] = meanMat[w-1][h]+meanMat[w][h-1]-meanMat[w-1][h-1];
                if(indexMat[w][h] != -1 && vFlagMat[w][h]){
                    tempV3d(0) =inputPCLptr->points[indexMat[w][h]].x;
                    tempV3d(1) =inputPCLptr->points[indexMat[w][h]].y;
                    tempV3d(2) =inputPCLptr->points[indexMat[w][h]].z;
                    meanMat[w][h] += tempV3d;
                    varianceMat[w][h] += tempV3d * tempV3d.transpose();
                }
            }
        }
    }
    inline double caculateD(PointI p1,PointI p2){
        return sqrt((p1.x -p2.x)*(p1.x - p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }
    inline double caculateNmN (const int& w1,const int& h1,const int& w2,const int& h2){
        int pointsNum1 = (boundMat[w1][h1].rightW - boundMat[w1][h1].leftW+1)*(boundMat[w1][h1].upH - boundMat[w1][h1].bottomH +1);
        int pointsNum2 = (boundMat[w2][h2].rightW - boundMat[w2][h2].leftW+1)*(boundMat[w2][h2].upH - boundMat[w2][h2].bottomH +1);
        if(pointsNum1 <6 || pointsNum2<6)
            return 0;
        Matrix3d variance1 = varianceMat[boundMat[w1][h1].rightW][boundMat[w1][h1].upH];
        Vector3d mean1 = meanMat[boundMat[w1][h1].rightW][boundMat[w1][h1].upH];
        Matrix3d variance2 = varianceMat[boundMat[w2][h2].rightW][boundMat[w2][h2].upH];
        Vector3d mean2 = meanMat[boundMat[w2][h2].rightW][boundMat[w2][h2].upH];
        if(boundMat[w1][h1].leftW !=0){
            variance1 -= varianceMat[boundMat[w1][h1].leftW - 1][boundMat[w1][h1].upH];
            mean1 -= meanMat[boundMat[w1][h1].leftW - 1][boundMat[w1][h1].upH];
        }
        if(boundMat[w1][h1].bottomH !=0){
            variance1 -= varianceMat[boundMat[w1][h1].rightW][boundMat[w1][h1].bottomH - 1];
            mean1 -= meanMat[boundMat[w1][h1].rightW][boundMat[w1][h1].bottomH - 1];

        }
        if(boundMat[w1][h1].leftW != 0 && boundMat[w1][h1].bottomH != 0){
            variance1 += varianceMat[boundMat[w1][h1].leftW -1][boundMat[w1][h1].bottomH -1];
            mean1 += meanMat[boundMat[w1][h1].leftW -1][boundMat[w1][h1].bottomH -1];
        }

        variance1 /= (boundMat[w1][h1].rightW - boundMat[w1][h1].leftW+1)*(boundMat[w1][h1].upH - boundMat[w1][h1].bottomH +1);
        mean1 = mean1/((boundMat[w1][h1].rightW - boundMat[w1][h1].leftW+1)*(boundMat[w1][h1].upH - boundMat[w1][h1].bottomH +1));
        variance1 -= mean1*mean1.transpose();


        if(boundMat[w2][h2].leftW !=0){
            variance2 -= varianceMat[boundMat[w2][h2].leftW - 1][boundMat[w2][h2].upH];
            mean2 -= meanMat[boundMat[w2][h2].leftW - 1][boundMat[w2][h2].upH];
        }
        if(boundMat[w2][h2].bottomH !=0){
            variance2 -= varianceMat[boundMat[w2][h2].rightW][boundMat[w2][h2].bottomH - 1];
            mean2 -= meanMat[boundMat[w2][h2].rightW][boundMat[w2][h2].bottomH - 1];

        }
        if(boundMat[w2][h2].leftW != 0 && boundMat[w2][h2].bottomH != 0){
            variance2 += varianceMat[boundMat[w2][h2].leftW -1][boundMat[w2][h2].bottomH -1];
            mean2 += meanMat[boundMat[w2][h2].leftW -1][boundMat[w2][h2].bottomH -1];
        }
        variance2 /= (boundMat[w2][h2].rightW - boundMat[w2][h2].leftW+1)*(boundMat[w2][h2].upH - boundMat[w2][h2].bottomH +1);
        mean2 = mean2/((boundMat[w2][h2].rightW - boundMat[w2][h2].leftW+1)*(boundMat[w2][h2].upH - boundMat[w2][h2].bottomH +1));
        variance2 -= mean2*mean2.transpose();

        EigenSolver<Matrix3d> es;
        es.compute(variance1);
        int va1 =0 ;
        for(int i=1;i<3;i++){
            if(es.eigenvalues()[i].real()<es.eigenvalues()[va1].real())
                va1 = i;
        }
        Vector3d ve1 = es.eigenvectors().col(va1).real();

        es.compute(variance2);
        int va2 =0 ;
        for(int i=1;i<3;i++){
            if(es.eigenvalues()[i].real()<es.eigenvalues()[va2].real())
                va2 = i;
        }
        Vector3d ve2 = es.eigenvectors().col(va2).real();
        double result = ve1.dot(ve2);
        return fabs(result);
    }
    void infectNearby(const int& w,const int& h,const int& lable){
        clusterMat[w][h] = lable;
        //infect left
        if(w > 0 && vFlagMat[w-1][h] && clusterMat[w-1][h] == 0 && indexMat[w-1][h] != -1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w-1][h]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w-1][h]])<DTRED || caculateNmN(w,h,w-1,h) > NTRED)
                infectNearby(w-1,h,lable);
        }
        //infect right
        if(w<WIDTH-1 && vFlagMat[w+1][h] && clusterMat[w+1][h] == 0 && indexMat[w+1][h] != -1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w+1][h]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w+1][h]])<DTRED || caculateNmN(w,h,w+1,h) > NTRED)
                infectNearby(w + 1,h,lable);
        }
        //infect down
        if(h>0 && vFlagMat[w][h-1] && clusterMat[w][h-1] == 0 && indexMat[w][h-1]!=-1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h-1]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h-1]])<DTRED || caculateNmN(w,h,w,h-1) > NTRED)
                infectNearby(w,h-1,lable);
        }
        //infect up
        if(h<HEIGHT-1 && vFlagMat[w][h+1] && clusterMat[w][h+1] == 0 && indexMat[w][h+1]!= -1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h+1]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h+1]])<DTRED || caculateNmN(w,h,w,h+1) > NTRED)
                infectNearby(w,h+1,lable);
        }

        //偏移两格
        //infect left
        if(w > 1 && vFlagMat[w-2][h] && clusterMat[w-2][h] == 0 && indexMat[w-2][h] != -1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w-2][h]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w-2][h]])<DTRED || caculateNmN(w,h,w-2,h) > NTRED)
                infectNearby(w-2,h,lable);
        }
        //infect right
        if(w<WIDTH-2 && vFlagMat[w+2][h] && clusterMat[w+2][h] == 0 && indexMat[w+2][h] != -1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w+2][h]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w+2][h]])<DTRED || caculateNmN(w,h,w+2,h) > NTRED)
                infectNearby(w + 2,h,lable);
        }
        //infect down
        if(h>1 && vFlagMat[w][h-2] && clusterMat[w][h-2] == 0 && indexMat[w][h-2]!=-1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h-2]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h-2]])<DTRED || caculateNmN(w,h,w,h-2) > NTRED)
                infectNearby(w,h-2,lable);
        }
        //infect up
        if(h<HEIGHT-2 && vFlagMat[w][h+2] && clusterMat[w][h+2] == 0 && indexMat[w][h+2]!= -1
                && caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h+2]])< 1){
            if(caculateD(inputPCLptr->points[indexMat[w][h]],inputPCLptr->points[indexMat[w][h+2]])<DTRED || caculateNmN(w,h,w,h+2) > NTRED)
                infectNearby(w,h+2,lable);
        }
    }
    void setClusterMap(){
        lable = 1;
        for(int w = 0; w< WIDTH;w++){
            for(int h = 0;h <HEIGHT;h++){
                if(vFlagMat[w][h] && clusterMat[w][h] == 0 && indexMat[w][h] != -1){
                    infectNearby(w,h, lable);
                    lable++;
                }
            }
        }
        // for(int w = 0 ;w < WIDTH ; w++){
            // cout << w << ':' ;
            // for(int h = 0;h < HEIGHT;h++){
                // cout << clusterMat[w][h] << ' ' ;
            // }
            // cout << endl;
        // }
    }
    //　每簇点云进行分类，使用奇数代表柱子，偶数代表墙
    void classsifyCluster(){
        Vector3d mean[lable+1];
        Matrix3d squareSum[lable+1];
        int number[lable+1];
        for(int i=0;i < lable+1;i++){
            mean[i] =  MatrixXd::Zero(3, 1);
            squareSum[i] = Matrix3d::Zero();
            number[i] = 0;
        }
        Vector3d tempP;
        for(int w=0;w < WIDTH ;w++){
            for(int h=0;h<HEIGHT;h++){
                 if(clusterMat[w][h] == 0 || indexMat[w][h] == -1 )
                     continue;
                 tempP(0) = inputPCLptr->points[indexMat[w][h]].x;
                 tempP(1) = inputPCLptr->points[indexMat[w][h]].y;
                 tempP(2) = inputPCLptr->points[indexMat[w][h]].z;
                 number[clusterMat[w][h]]++;
                 mean[clusterMat[w][h]] += tempP;
                 squareSum[clusterMat[w][h]] += tempP*tempP.transpose();
            }
        }
        //太少点的点云簇会被删除
        bool isClusterFlag[lable+1] ;
        isClusterFlag[0] = false;
        const int numThreshold = 50;
        for(int i =1;i<lable+1;i++){
            isClusterFlag[i] = number[i] < numThreshold ? false : true;
        }

        //用奇数代表柱子，偶数代表墙面
        int lineNum = 1,planeNum = 2;
        int lableAddlp[lable+1];
        const double el = 0.1,ep = 0.1; //paper eq7 and eq8
        for(int i =1;i<lable+1;i++){
            if(!isClusterFlag[i])
                continue;
            mean[i] = mean[i]/ number[i];
            Matrix3d variance = squareSum[i] / number[i] - mean[i]*mean[i].transpose();
            EigenSolver<Matrix3d> es;
            es.compute(variance);
            char e1 = 0,e2 = 1,e3 = 2;//index of eigenvalue, small ,middle,big
            if(es.eigenvalues()[e1].real() > es.eigenvalues()[e2].real()){
                char tc = e1;
                e1 = e2;
                e2 = tc;
            }
            if(es.eigenvalues()[e1].real() > es.eigenvalues()[e3].real()){
                char tc = e1;
                e1 = e3;
                e3 = tc;
            }
            if(es.eigenvalues()[e2].real() > es.eigenvalues()[e3].real()){
                char tc = e2;
                e2 = e3;
                e3 = tc;
            }
            double ev[3] = {es.eigenvalues()[e1].real(), es.eigenvalues()[e2].real(), es.eigenvalues()[e3].real()};
            if((ev[0]+ev[1])/(ev[0]+ev[1]+ev[2]) < el){
                lableAddlp[i] = lineNum + i;
                // cout << lineNum << ':' ;
                lineNum += 2;
            }else if(ev[0] / (ev[0]+ev[1]+ev[2]) < ep){
                lableAddlp[i] = planeNum + i;
                // cout << planeNum << ':';
                planeNum += 2;
            }else{
                isClusterFlag[i] = false;
                // cout << "neither:";
            }
            // cout << ev[0] <<' '<< ev[1] << ' ' << ev[2] << endl;
            // cout << es.eigenvalues().real().transpose() << endl;
            // cout << es.eigenvectors().real()<< endl;
        }
        for(int w = 0;w< WIDTH;w++){
            for(int h = 0;h< HEIGHT ;h++){
                if(isClusterFlag[clusterMat[w][h]]){
                    clusterMat[w][h] = lableAddlp[clusterMat[w][h]] - clusterMat[w][h];
                }else{
                    clusterMat[w][h] = 0;
                }
            }
        }
        for(int w = 0 ;w < WIDTH ; w++){
            // cout << w << ':' ;
            for(int h = 0;h < HEIGHT;h++){
                // cout << clusterMat[w][h] << ' ' ;
                // if(clusterMat[w][h] != 0 && clusterMat[w][h]%2 == 1){
                if(clusterMat[w][h] != 0){
                // if(vFlagMat[w][h]){
                    planeAndLinePCLptr->push_back(inputPCLptr->points[indexMat[w][h]]);
                }
            }
            // cout << endl;
        }
        // cout << planeAndLinePCLptr->size() << endl;
        sensor_msgs::PointCloud2 msg_pcl2;
        pcl::toROSMsg(*planeAndLinePCLptr,msg_pcl2);
        msg_pcl2.header.frame_id = frameID;
        msg_pcl2.header.stamp = ros::Time::now();
        planeAndLine_pub.publish(msg_pcl2);
    }
    void computeNormal(){
        for(int w =0 ;w< WIDTH;w++){
            for(int h = 0;h < HEIGHT;h++){
                if(clusterMat[w][h] == 0 || clusterMat[w][h] %2 ==0 )
                    continue;
                Matrix3d variance = varianceMat[boundMat[w][h].rightW][boundMat[w][h].upH];
                Vector3d mean = meanMat[boundMat[w][h].rightW][boundMat[w][h].upH];
                if(boundMat[w][h].leftW !=0){
                    variance -= varianceMat[boundMat[w][h].leftW - 1][boundMat[w][h].upH];
                    mean -= meanMat[boundMat[w][h].leftW - 1][boundMat[w][h].upH];
                }
                if(boundMat[w][h].bottomH !=0){
                    variance -= varianceMat[boundMat[w][h].rightW][boundMat[w][h].bottomH - 1];
                    mean -= meanMat[boundMat[w][h].rightW][boundMat[w][h].bottomH - 1];

                }
                if(boundMat[w][h].leftW != 0 && boundMat[w][h].bottomH != 0){
                    variance += varianceMat[boundMat[w][h].leftW -1][boundMat[w][h].bottomH -1];
                    mean += meanMat[boundMat[w][h].leftW -1][boundMat[w][h].bottomH -1];
                }
                variance /= (boundMat[w][h].rightW - boundMat[w][h].leftW+1)*(boundMat[w][h].upH - boundMat[w][h].bottomH +1);
                mean = mean/((boundMat[w][h].rightW - boundMat[w][h].leftW+1)*(boundMat[w][h].upH - boundMat[w][h].bottomH +1));
                variance -= mean*mean.transpose();
                EigenSolver<Matrix3d> es;
                es.compute(variance);
                int va =0 ;
                for(int i=1;i<3;i++){
                    if(es.eigenvalues()[i].real()<es.eigenvalues()[va].real())
                        va = i;
                }
                Vector3d ve = es.eigenvectors().col(va).real();
                PointNormal point;
                point.x = inputPCLptr->points[indexMat[w][h]].x;
                point.y = inputPCLptr->points[indexMat[w][h]].y;
                point.z = inputPCLptr->points[indexMat[w][h]].z;
                point.normal[0] = ve[0];
                point.normal[1] = ve[1];
                point.normal[2] = ve[2];
                normalPCLptr->push_back(point);
            }
        }
    }
    ~PlaneAndLineDetect(){
    }
private:
    inline void getPitchAngle(PointCloud<PointI>::Ptr cloudIn){
        std::vector<double> angle;
        for(int i=0;i<cloudIn->size();i++){
            // std::cout << cloudIn->points[i].intensity <<' ';
            // max = max > cloudIn->points[i].intensity ? max : cloudIn->points[i].intensity;
            // min = min < cloudIn->points[i].intensity ? min : cloudIn->points[i].intensity;
            double x = cloudIn->points[i].x;
            double y = cloudIn->points[i].y;
            double z = cloudIn->points[i].z;
            double a = atan2(z,sqrt(x*x+y*y));
            // double a = (z/sqrt(x*x+y*y));
            if(!(a==a))
                continue;
            if(angle.size() ==0){
                angle.push_back(a);
            }else{
                bool flag = true;
                for(int j=0;j<angle.size();j++){
                    if(a - angle[j] < 0.0001 && a - angle[j] > -0.0001){
                        flag = false;
                        break;
                    }
                }
                if(flag){
                    angle.push_back(a);
                }
            }
        }

        cout << endl<< "angle size = " <<  angle.size() << endl;
        for(int i=0;i<angle.size() ;i++){
            cout << angle[i] << endl;
        }
    }
public:
    PointCloud<PointI>::Ptr inputPCLptr;
    PointCloud<PointI>::Ptr planeAndLinePCLptr;
    PointCloud<PointNormal>::Ptr normalPCLptr;
    vector<PointCloud<PointI>::Ptr> resultPCLptr;
    double depthMat[WIDTH][HEIGHT] = {{0}}; //D mat in paper
    int indexMat[WIDTH][HEIGHT] = {{-1}};  //I mat paper
    bool vFlagMat[WIDTH][HEIGHT] ;
    Bound boundMat[WIDTH][HEIGHT];
    Vector3d meanMat[WIDTH][HEIGHT];
    Matrix3d varianceMat[WIDTH][HEIGHT];
    int clusterMat[WIDTH][HEIGHT];
    int lable;

    ros::NodeHandle nh;
    string frameID;

    ros::Publisher planeAndLine_pub;
    ros::Subscriber lidarpoints_sub;
private:
    // static constexpr double cos[HEIGHT] =    {-0.261789,
    //                                      -0.226484,
    //                                      -0.191801,
    //                                      -0.156643,
    //                                      -0.122323,
    //                                      -0.0870919,
    //                                      -0.0522656,
    //                                      -0.0173102,
    //                                      0.0176854,
    //                                      0.0526391,
    //                                      0.0874637,
    //                                      0.122077,
    //                                      0.15701,
    //                                      0.192162,
    //                                      0.226247,
    //                                      0.261555,
    //                                      }; //from bottom to top hudu
};

#endif  //  PLANE_AND_LINE_DETECT
