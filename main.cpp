#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>


template<class T>
void deletePtr(T *ptr , const char *ptrname , const char *funname , const int line)
{
    if (!ptr)
        delete ptr;

     std::cout << "位于" << funname << "函数中的第" << line << "行的指针：" << ptrname << "被释放成功了.." << std::endl;
}

// ! ==== [opencv_test1()]
void createAlphaMat(cv::Mat *mat)
{
    CV_Assert(mat->channels() == 4);
    for (int i = 0; i < mat->rows ; ++i){
        for (int j = 0; j < mat->cols ; ++j){
            /*
             * cv::Vec4b 此模板类表示可以执行基本算术运算的简单数字向量（1,2,3,4 ...元素）
             * cv::at(i , j) 返回指定数组元素的引用
             * cv::saturate_cast<uchar> 类似于标准的C++转换操作,表示从一个原始类型到另一个原始类型的有效和准确的转换，
             *                      当输入值超出目标类型的范围时，则剪切该值
             */
            cv::Vec4b &bgra = mat->at<cv::Vec4b>(i , j);
            bgra[0] = UCHAR_MAX; //Blue
            bgra[1] = cv::saturate_cast<uchar>( (float (mat->cols - j)) / ( (float)mat->cols) * UCHAR_MAX); //Green
            bgra[2] = cv::saturate_cast<uchar>( (float (mat->rows - i)) / ( (float)mat->rows) * UCHAR_MAX); //Red
            bgra[3] = cv::saturate_cast<uchar>( 0.5 * (bgra[1] + bgra[2])); //Alpha
        }
    }
}
void opencv_test1()
{
//    cv::Mat R(3, 2, CV_8UC3);
//    cv::randu(R, cv::Scalar::all(0), cv::Scalar::all(255)); //产生随机矩阵值
//    std::cout << "R = " << std::endl << " " << R << std::endl;

//    cv::Mat m(2 , 3 , CV_8UC3 , cv::Scalar(0 , 0 , 255));
//    std::cout << "m = " << std::endl << " " << m << std::endl;
    cv::Mat *mat = new cv::Mat(480 , 640 , CV_8UC4);
    createAlphaMat(mat);

    std::vector<int> *compression_params = new std::vector<int>();
    compression_params->push_back(cv::IMWRITE_PNG_COMPRESSION); //cv::IMWRITE_PNG_COMPRESSION png图片压缩级别
    compression_params->push_back(9);


    try{
        cv::imwrite("./alpha.png" , *mat , *compression_params);
    }
    catch (cv::Exception &ex){
        std::fprintf(stderr , "Exception convertion image to PNG format: %s\n" , ex.what());
        return;
    }

    std::fprintf(stdout , "Saved PNG file with alpha data .\n");

    cv::imshow("./alpha.png" , *mat);
    cv::waitKey(0);

    delete compression_params;
    delete mat;

}
// ! ==== [opencv_test1()]

// ! ===== [opencv_test2()]
void opencv_test2(const int *argc ,  char **argv)
{
     char *imageName = argv[1];

    cv::Mat *image = new cv::Mat();
    *image = cv::imread(imageName , cv::IMREAD_COLOR);
    if (*argc != 2 || !image->data)
    {
        std::cout << "imageName 可能是一个无效的图片路径" << std::endl;
        delete image;
        return ;
    }

    cv::Mat *gray_image = new cv::Mat();
    cv::cvtColor(*image , *gray_image , cv::COLOR_BGR2GRAY); //彩色转灰色
    cv::imwrite("./gray_image.jpg" , *gray_image);

    cv::namedWindow(imageName , cv::WINDOW_AUTOSIZE);
    cv::namedWindow("gray image" , cv::WINDOW_AUTOSIZE);

    cv::imshow(imageName , *image);
    cv::imshow("gray image" , *gray_image);
    cv::waitKey(0);


    delete gray_image;
    delete image;
}
// ! ===== [opencv_test2()]

// ! ===== [opencv_test3()]
void opencv_test3()
{
    cv::Point2f *p = new cv::Point2f(5 ,1);
    std::cout << "Point2f (2D) = " << *p << std::endl;

    std::cout << "====================" << std::endl;

    std::vector<float> *v = new std::vector<float>();
    v->push_back( (float)CV_PI );
    v->push_back(2);
    v->push_back(3.01f);
    std::cout << "Vector of floats via Mat = " << std::endl << cv::Mat(*v) << std::endl;

    std::cout << "====================" << std::endl;

    std::vector<cv::Point2f> vPoints(20);
    for (size_t i = 0 ; i < vPoints.size() ; ++i){
        vPoints[i] = cv::Point2f( (float)(i * 5) , (float)(i % 7) );
    }
    std::cout << "A vector of 2D Points = " << std::endl << vPoints << std::endl;


    delete p;
    delete v;
}
// ! ===== [opencv_test3()]

// ! ================ [opencv_test4()]
cv::Mat& ScanImageAndReduceC(cv::Mat& I, const uchar* table);
cv::Mat& ScanImageAndReduceIterator(cv::Mat& I, const uchar* table);
cv::Mat& ScanImageAndReduceRandomAccess(cv::Mat& I, const uchar * table);
void opencv_test4(const int *argc , char **argv)
{

    if (*argc < 3){
        std::cout << "Not enough parameters" << std::endl;
        return ;
    }


    cv::Mat I, J;
        if( *argc == 4 && !strcmp(argv[3],"G") )
            I = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        else
            I = cv::imread(argv[1], cv::IMREAD_COLOR);

        if (I.empty())
        {
            std::cout << "The image" << argv[1] << " could not be loaded." << std::endl;
            return ;
        }

        //! [dividewith]
        int divideWith = 0; // convert our input string to number - C++ style
        std::stringstream s;
        s << argv[2];
        s >> divideWith;
        if (!s || !divideWith)
        {
            std::cout << "Invalid number entered for dividing. " << std::endl;
            return ;
        }

        uchar table[256];
        for (int i = 0; i < 256; ++i)
           table[i] = (uchar)(divideWith * (i/divideWith));
        //! [dividewith]

        const int times = 100;
        std::chrono::duration<double> t;


        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();


        for (int i = 0; i < times; ++i)
        {
            cv::Mat clone_i = I.clone();
            J = ScanImageAndReduceC(clone_i, table);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        t = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1);


        std::cout << "Time of reducing with the C operator [] (averaged for "
             << times << " runs): " << t.count() << " milliseconds."<< std::endl;


        t1 = std::chrono::steady_clock::now();

        for (int i = 0; i < times; ++i)
        {
            cv::Mat clone_i = I.clone();
            J = ScanImageAndReduceIterator(clone_i, table);
        }



        t2 = std::chrono::steady_clock::now();
        t = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1);
        std::cout << "Time of reducing with the iterator (averaged for "
            << times << " runs): " << t.count() << " milliseconds."<< std::endl;



        t1 = std::chrono::steady_clock::now();

        for (int i = 0; i < times; ++i)
        {
            cv::Mat clone_i = I.clone();
            ScanImageAndReduceRandomAccess(clone_i, table);
        }



        t2 = std::chrono::steady_clock::now();
        t = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1);
        std::cout << "Time of reducing with the on-the-fly address generation - at function (averaged for "
            << times << " runs): " << t.count() << " milliseconds."<< std::endl;

        //! [table-init]
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for( int i = 0; i < 256; ++i)
            p[i] = table[i];
        //! [table-init]


        t1 = std::chrono::steady_clock::now();

        for (int i = 0; i < times; ++i)
            //! [table-use]
            LUT(I, lookUpTable, J);
            //! [table-use]


        t2 = std::chrono::steady_clock::now();
        t = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1);
        std::cout << "Time of reducing with the LUT function (averaged for "
            << times << " runs): " << t.count() << " milliseconds."<< std::endl;


        return;
}

//! [scan-c]
cv::Mat& ScanImageAndReduceC(cv::Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    int nRows = I.rows;
    int nCols = I.cols * channels;

    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i); //返回指向指定矩阵行的指针
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}
//! [scan-c]

//! [scan-iterator]
cv::Mat& ScanImageAndReduceIterator(cv::Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            cv::MatIterator_<uchar> it, end;
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                *it = table[*it];
            break;
        }
    case 3:
        {
            cv::MatIterator_<cv::Vec3b> it, end;
            for( it = I.begin<cv::Vec3b>(), end = I.end<cv::Vec3b>(); it != end; ++it)
            {
                (*it)[0] = table[(*it)[0]];
                (*it)[1] = table[(*it)[1]];
                (*it)[2] = table[(*it)[2]];
            }
        }
    }

    return I;
}
//! [scan-iterator]

//! [scan-random]
cv::Mat& ScanImageAndReduceRandomAccess(cv::Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
            break;
        }
    case 3:
        {
         cv::Mat_<cv::Vec3b> _I = I;

         for( int i = 0; i < I.rows; ++i)
            for( int j = 0; j < I.cols; ++j )
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;
         break;
        }
    }

    return I;
}
//! [scan-random]

// ! ================ [opencv_test4()]



// ! ========== [opencv_test5()]
void Sharpen(const cv::Mat& myImage,cv::Mat& Result);

void opencv_test5(const int *argc , char **argv)
{
    std::cout << " usage: [image_name -- default ../data/lena.jpg] [G -- grayscale] " << std::endl;

    const char *filename = *argc >= 2 ? argv[1] : "../data/lena.jpg";

    cv::Mat *src = new cv::Mat();
    cv::Mat *dst0 = new cv::Mat();
    cv::Mat *dst1 = new cv::Mat();

    if (*argc >= 3 && !strcmp("G" , argv[2])){
        *src = cv::imread(filename , cv::IMREAD_GRAYSCALE);
    }
    else
        *src = cv::imread(filename , cv::IMREAD_COLOR);

    if (src->empty()){
        std::cout << "Can't open image [" << filename << "]" << std::endl;
        std::cout << "异常退出时，记得释放指针！" << std::endl;
        deletePtr(src , "src" , __FUNCTION__ , __LINE__);
        deletePtr(dst0 , "dst0" , __FUNCTION__ , __LINE__);
        deletePtr(dst1 , "dst1" , __FUNCTION__ , __LINE__);
        return ;
    }


    cv::namedWindow("Input" , cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output" , cv::WINDOW_AUTOSIZE);

    cv::imshow("Input" , *src);
    double t = (double)cv::getTickCount(); //返回每秒钟的滴答数

    // !---------------------------------- [dst0]
    Sharpen(*src , *dst0);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); //计算执行所用的时间
    std::cout << "Hand writeen function times passed in seconds: " << t << std::endl;
    cv::imshow("Output" , *dst0);
    //cv::waitKey(0);

    // !------------------------------------[dst0]

    cv::Mat kernel = (cv::Mat_<char>(3 , 3) << 0 , -1 , 0 ,
                                                 -1 , 5 , -1 ,
                                                 0 , -1 , 0 );
    t = (double)cv::getTickCount();


    // !-------------------------------- [filter2D]
    cv::filter2D(*src , *dst1 , src->depth() , kernel);
    t = ( (double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Built-in fileter2D time passed in seconds: " << t << std::endl;

    cv::imshow("Output" , *dst1);
    cv::waitKey(0);

    // !-------------------------------- [filter2D]

    std::cout << "正常退出时，记得释放指针！" << std::endl;
    deletePtr(src , "src" , __FUNCTION__ , __LINE__);
    deletePtr(dst0 , "dst0" , __FUNCTION__ , __LINE__);
    deletePtr(dst1 , "dst1" , __FUNCTION__ , __LINE__);
}

void Sharpen(const cv::Mat &myImage, cv::Mat &Result)
{
    // ! [8_bit]
    CV_Assert(myImage.depth() == CV_8U);
    // ! [8_bit]

    // ! [creat_channerls]
    const int nChannels = myImage.channels();
    Result.create(myImage.size() , myImage.type());
    // ! [creat_channerls]

    // ! ----- [basic_method_loop]
    for (int j = 1 ; j < myImage.rows - 1 ; ++j){
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current = myImage.ptr<uchar>(j);
        const uchar* next = myImage.ptr<uchar>(j + 1);

        uchar *output = Result.ptr<uchar>(j);

        for (int i = nChannels ; i < nChannels * (myImage.cols - 1) ; ++i){
            *output = static_cast<uchar>( 5 * current[i] - current[i - nChannels]
                    - current[i + nChannels] - previous[i] - next[i]);
        }
    }
    // ! ----- [basic_method_loop]


    // ! [borders]
    Result.row(0).setTo(cv::Scalar(0));
    Result.row( Result.rows - 1).setTo(cv::Scalar(0));
    Result.col(0).setTo(cv::Scalar(0));
    Result.col( Result.cols - 1).setTo(cv::Scalar(0));
    // ! [borders]


}

// ! ========== [opencv_test5()]

int main(int argc , char *argv[])
{
    //opencv_test1();  // 将cv::Mat 写入文件
    //opencv_test2(&argc , argv);  //彩色图片转灰色图片
    //opencv_test3();  //点的操作 point
    //opencv_test4(&argc , argv);   //扫描图片 , 最快的扫描图片方法是cv::LUT()函数 ，在教程中的介绍

    opencv_test5(&argc , argv);  //矩阵上的掩码操作

    return 0;
}
