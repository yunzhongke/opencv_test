#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include "common/common.h"

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

// ! ========== [opencv_test6()]
void opencv_test6(const int *argc , char **argv)
{
    if (*argc < 2){
        std::cout << " usage: [program] [image_name] " << std::endl;
        return;
    }

    //! [3 channel imagel]
    cv::Mat *img = new cv::Mat();
    *img = cv::imread(argv[1]);
    //cv::Vec3b *intensity = img->at<cv::Vec3b>(cv::Point(x , y));
    cv::MatIterator_<cv::Vec3b> it;
    for (it = img->begin<cv::Vec3b>() ; it != img->end<cv::Vec3b>() ; ++it){
        std::cout << "B : " << (*it)[0] << " , G : " << (*it)[1] << " , R : " << (*it)[2] << std::endl;
//        (*it)[0] = (*it)[1];
//        (*it)[1] = (*it)[2];
//        (*it)[2] = (*it)[0];
    }
    cv::imshow("ouput1" , *img);
    cv::waitKey(0);
    //! [3 channel imagel]

    //! [a single channel grey image]
    cv::Mat *grey = new cv::Mat();
    *grey = cv::imread(argv[1] , cv::IMREAD_GRAYSCALE);
    cv::MatIterator_<uchar> itg;
    for (itg = grey->begin<uchar>() ; itg != grey->end<uchar>() ; ++itg){
        std::cout << " grey : " << *itg << std::endl;
//        *itg *= 2;
    }
    cv::imshow("output2" , *grey);
    cv::waitKey(0);
    //! [a single channel grey image]

    //! [CV_32 -> CV_8U]
    cv::Mat *img2 = new cv::Mat();
    *img2 = cv::imread(argv[1]);
    cv::Mat grey2;
    cv::cvtColor(*img2 , grey2 , cv::COLOR_BGR2BGRA);
    cv::Mat sobelx;
    cv::Sobel(grey2 , sobelx , CV_32F , 1 , 0);
    double minVal , maxVal;
    cv::minMaxLoc(sobelx , &minVal , &maxVal);
    cv::Mat draw;
    sobelx.convertTo(draw , CV_8U , 255.0 / (maxVal - minVal) , - minVal * 255.0 / (maxVal - minVal) );
    cv::imshow("output3" , draw);
    cv::waitKey(0);
    //! [CV_32 -> CV_8U]

    std::cout << "正常退出时，记得释放指针！" << std::endl;
    deletePtr(img , "img" , __FUNCTION__ , __LINE__);
    deletePtr(grey , "grey" , __FUNCTION__ , __LINE__);
    deletePtr(img2 , "img2" , __FUNCTION__ , __LINE__);
}
// ! ========== [opencv_test6()]

// ! ========== [opencv_test7()]
void opencv_test7(const int *argc , char **argv)
{
    if (*argc != 3){
        std::cout << "usage : [program] [image_name1] [image_name2] " << std::endl;
        return ;
    }

    double beta , input , alpha = 0.5;
    cv::Mat *image = new cv::Mat();
    cv::Mat *new_image = new cv::Mat();
    cv::Mat *dst = new cv::Mat();
    std::cout << " Simple Linear Blender " << std::endl;
    std::cout << " ------------------- " << std::endl;
    std::cout << "* Enter alpha [0-1]: " ;
    std::cin >> input;

    if (input >= 0 && input <= 1){
        alpha = input ;
    }
    *image = cv::imread(argv[1]);
    *new_image = cv::imread(argv[2]);
    if (image->empty() || new_image->empty()){
        std::cout << "Error loading image or new_image " << std::endl;
        deletePtr(image , "image" , __FUNCTION__ , __LINE__);
        deletePtr(new_image , "new_image" , __FUNCTION__ , __LINE__);
        deletePtr(dst , "dst" , __FUNCTION__ , __LINE__);
        return ;
    }
    beta = (1.0 - alpha);
    cv::addWeighted(*image , alpha , *new_image , beta , 0.0 , *dst);
    cv::imshow("Linear Blend" , *dst);
    cv::waitKey(0);
    std::cout << "正常退出时，记得释放指针！" << std::endl;
    deletePtr(image , "image" , __FUNCTION__ , __LINE__);
    deletePtr(new_image , "new_image" , __FUNCTION__ , __LINE__);
    deletePtr(dst , "dst" , __FUNCTION__ , __LINE__);
}
// ! ========== [opencv_test7()]

// ! ========== [opencv_test8()]
void basicLinearTransform(const cv::Mat *img , cv::Mat **out_img , const double alpha_ , const int beta_);
void on_linear_transform_alpha_trackbar( int , void *params);
void on_linear_transform_beta_trackbar( int , void *params);
void on_gamma_correction_trackbar( int , void *params);
void gammaCorrection(const cv::Mat *img , cv::Mat **out_img , const double gamma_);

struct Params
{
    cv::Mat *img_original;
    cv::Mat *img_corrected;
    cv::Mat *img_gamma_corrected;
    int alpha_;
    int beta_;
    int gamma_cor_;
};  // 传参用
void opencv_test8(const int *argc , char **argv)
{

    if (*argc != 2){
        std::cout << "usage : [program] [image]" << std::endl;
        return;
    }

    int alpha = 100 , beta = 100 , gamma_cor = 100;

    Params *param1 = new Params();
    param1->img_original = new cv::Mat();
    param1->img_corrected = new cv::Mat();
    param1->img_gamma_corrected = new cv::Mat();
    *param1->img_original = cv::imread(argv[1]);
    *param1->img_corrected = cv::Mat(param1->img_original->rows , param1->img_original->cols * 2 , param1->img_original->type());
    *param1->img_gamma_corrected = cv::Mat(param1->img_original->rows , param1->img_original->cols * 2 , param1->img_original->type());


    cv::hconcat(*param1->img_original , *param1->img_original , *param1->img_corrected);  // cv::hconcat() 代表将多个矩阵水平连接成一个矩阵
    cv::hconcat(*param1->img_original , *param1->img_original , *param1->img_gamma_corrected);

    cv::namedWindow( "Brightness and contrast adjustments", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gamma correction" , cv::WINDOW_AUTOSIZE);

    param1->alpha_ = alpha;
    param1->beta_ = beta;
    param1->gamma_cor_ = gamma_cor;
    cv::createTrackbar("Alpha gain (contrast)" , "Brightness and contrast adjustments" , &param1->alpha_ , 500 , on_linear_transform_alpha_trackbar , (void*)param1);
    cv::createTrackbar("Beta bias (brightness) " , "Brightness and contrast adjustments" , &param1->beta_ , 200 , on_linear_transform_beta_trackbar , (void*)param1);
    cv::createTrackbar("Gamma correction" , "Gamma correction" , &param1->gamma_cor_ , 200 , on_gamma_correction_trackbar , (void*)param1);

    while (true){
        cv::imshow("Brightness and contrast adjustments" , *param1->img_corrected);
        cv::imshow("Gamma correction" , *param1->img_gamma_corrected);

        int c = cv::waitKey(20);
        if (c == 27)
            break;
    }

    cv::imwrite("linear_transform_correction.png" , *param1->img_corrected);
    cv::imwrite("gamma_correction.png" , *param1->img_gamma_corrected);

    std::cout << "正常退出时，记得释放指针！" << std::endl;
    if (param1){
        delete param1->img_corrected;
        delete param1->img_gamma_corrected;
        delete param1->img_original;
    }
    delete param1;
}

void basicLinearTransform(const cv::Mat *img, cv::Mat **out_img , const double alpha_, const int beta_)
{
    cv::Mat res;

    img->convertTo(res , -1 , alpha_ , beta_);
    cv::hconcat(*img , res , **out_img);
}
void gammaCorrection(const cv::Mat *img, cv::Mat **out_img , const double gamma_)
{
    CV_Assert(gamma_ >= 0);

    //![changing-contrast-brightness-gamma-correction]
    cv::Mat lookUpTable(1 , 256 , CV_8U);
    uchar *p = lookUpTable.ptr();
    for ( int i = 0; i < 256; ++i){
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0 , gamma_) * 255.0);
    }
    cv::Mat res = img->clone();
    cv::LUT(*img , lookUpTable , res);
    //![changing-contrast-brightness-gamma-correction]

    cv::hconcat(*img , res , **out_img);
}
void on_linear_transform_alpha_trackbar(int , void *params)
{
    Params *ptr = (Params *)params;
    double alpha_value = ptr->alpha_ / 100.0;
    int beta_value = ptr->beta_ - 100;
    basicLinearTransform(ptr->img_original , &ptr->img_corrected , alpha_value , beta_value);
}
void on_linear_transform_beta_trackbar(int, void *params)
{
    Params *ptr = (Params *)params;
    double alpha_value = ptr->alpha_ / 100.0;
    int beta_value = ptr->beta_ - 100;
    basicLinearTransform(ptr->img_original , &ptr->img_corrected , alpha_value , beta_value);
}
void on_gamma_correction_trackbar(int, void *params)
{
    Params *ptr = (Params *)params;
    double gamma_value = ptr->gamma_cor_ / 100.0;
    gammaCorrection(ptr->img_original , &ptr->img_gamma_corrected , gamma_value);
}
// ! ========== [opencv_test8()]

// ! ========== [opencv_test9()]
void MyEllipse(cv::Mat *img , double angle);
void MyFilledCircle(cv::Mat *img , cv::Point center);
void MyLine(cv::Mat *img , cv::Point start , cv::Point end);
void MyPolygon(cv::Mat *img);
void opencv_test9()
{
   char atom_window[] = "Drawing 1: Atom";
   char rook_window[] = "Drawing 2: Rook";

   cv::Mat *atom_image = new cv::Mat();
   cv::Mat *rook_image = new cv::Mat();
   *atom_image = cv::Mat::zeros(w , w , CV_8UC3);
   *rook_image = cv::Mat::zeros(w , w , CV_8UC3);

   MyEllipse(atom_image , 90);
   MyEllipse(atom_image , 0);
   MyEllipse(atom_image , 45);
   MyEllipse(atom_image , -45);

   MyFilledCircle(atom_image , cv::Point(w/2 , w/2));

   MyPolygon(rook_image);
   cv::rectangle(*rook_image , cv::Point( 0 , 7 * w / 8) ,
                 cv::Point(w , w) , cv::Scalar( 0 ,255 , 255)
                 , cv::FILLED , cv::LINE_8);

   MyLine(rook_image , cv::Point(0 , 25 * w / 16) , cv::Point(w , 15 * w /16));
   MyLine(rook_image , cv::Point(w / 4 , 7 * w / 8) , cv::Point(w / 4 , w));
   MyLine(rook_image , cv::Point(w /2 , 7 * w / 8) , cv::Point(w / 2 , w));
   MyLine(rook_image , cv::Point(3 * w / 4 , 7 * w / 8) , cv::Point(3 * w / 4 , w));

   cv::imshow(atom_window , *atom_image);
   cv::moveWindow(atom_window , 0 ,200);
   cv::imshow(rook_window , *rook_image);
   cv::moveWindow(rook_window , w , 200);
   cv::waitKey(0);

   std::cout << "程序正常退出 ， 记得释放指针!" << std::endl;
   deletePtr(atom_image , "atom_image" , __func__ , __LINE__);
   deletePtr(rook_image , "rook_image" , __func__ , __LINE__);
}

void MyEllipse(cv::Mat *img, double angle)
{
    int thickness = 2;
    int lineType = 8;

    cv::ellipse(*img , cv::Point( w / 2 , w /2) , cv::Size( w / 4 , w / 16)
                , angle , 0 , 360 , cv::Scalar( 255 , 0 , 0) ,
                thickness , lineType );
}

void MyFilledCircle(cv::Mat *img, cv::Point center)
{
    cv::circle(*img ,
               center ,
               w / 32 ,
               cv::Scalar(0 , 0 ,255) ,
               cv::FILLED ,
               cv::LINE_8);
}

void MyPolygon(cv::Mat *img)
{
    int lineType = cv::LINE_8;
    cv::Point rook_points[1][20];

    rook_points[0][0]  = cv::Point(  w / 4 ,  7 * w / 8 );
    rook_points[0][1]  = cv::Point(  3 * w / 4,   7 * w / 8 );
    rook_points[0][2]  = cv::Point(  3 * w / 4,  13 * w / 16 );
    rook_points[0][3]  = cv::Point( 11 * w / 16, 13 * w / 16 );
    rook_points[0][4]  = cv::Point( 19 * w / 32,  3 * w / 8 );
    rook_points[0][5]  = cv::Point(  3 * w / 4,   3 * w / 8 );
    rook_points[0][6]  = cv::Point(  3 * w / 4,     w / 8 );
    rook_points[0][7]  = cv::Point( 26 * w / 40,    w / 8 );
    rook_points[0][8]  = cv::Point( 26 * w / 40,    w / 4 );
    rook_points[0][9]  = cv::Point( 22 * w / 40,    w / 4 );
    rook_points[0][10] = cv::Point( 22 * w / 40,    w / 8 );
    rook_points[0][11] = cv::Point( 18 * w / 40,    w / 8 );
    rook_points[0][12] = cv::Point( 18 * w / 40,    w / 4 );
    rook_points[0][13] = cv::Point( 14 * w / 40,    w / 4 );
    rook_points[0][14] = cv::Point( 14 * w / 40,    w / 8 );
    rook_points[0][15] = cv::Point(      w / 4,     w / 8 );
    rook_points[0][16] = cv::Point(      w / 4,   3 * w / 8 );
    rook_points[0][17] = cv::Point( 13 * w / 32,  3 * w / 8 );
    rook_points[0][18] = cv::Point(  5 * w / 16, 13 * w / 16 );
    rook_points[0][19] = cv::Point(    w / 4,  13 * w / 16 );

    const cv::Point *ppt[1] = {&rook_points[0][0]} ;
    int npt = 20;

    cv::fillPoly(*img , ppt , &npt , 1 , cv::Scalar( 255 , 2555 , 255) , lineType);
}

void MyLine(cv::Mat *img, cv::Point start, cv::Point end)
{
    int thickness = 2;
    int lineType = cv::LINE_8;

    cv::line(*img ,
             start ,
             end ,
             cv::Scalar(0 , 0 , 0) ,
             thickness ,
             lineType);
}
// ! ========== [opencv_test9()]

// ! ========== [opencv_test10()]
void opencv_test10()
{
    cv::Mat *img = new cv::Mat(400 , 400 , CV_8UC3 , cv::Scalar(255 , 255 , 255));
    cv::line(*img , cv::Point(20 , 40) , cv::Point(120 , 140) , cv::Scalar(255 , 0 , 0) , 3);
    cv::rectangle(*img , cv::Point(150 , 40) , cv::Point(250 , 140) , cv::Scalar(0 , 0 , 255) , -1);
    cv::circle(*img , cv::Point(330 , 90) , 50 , cv::Scalar(0 , 255 , 0) , -1);
    cv::ellipse(*img , cv::Point(80 , 280) , cv::Size(60 , 40) , 45 , 0 ,360 , cv::Scalar(255 , 255 , 0) , 2);
    cv::putText(*img , std::string("nihao") , cv::Point(150 , 80) , 0 , 1 , cv::Scalar(0 , 0 , 0));

    cv::Point points[1][5];
    const cv::Point *p[5];
    points[0][0] = cv::Point(150, 270);
    points[0][1] = cv::Point(190, 220);
    points[0][2] = cv::Point(260, 255);
    points[0][3] = cv::Point(224, 296);
    points[0][4] = cv::Point(178, 316);
    p[0] = &points[0][0];
    const int npts = 3;
    cv::polylines(*img , p , &npts , 1 , true , cv::Scalar( 255 , 0 , 0) , 5);

    cv::imshow("window" , *img);
    cv::waitKey(0);

   std::cout << "程序正常退出 ， 记得释放指针!" << std::endl;
   deletePtr(img , "img" , __func__ , __LINE__);
}
// ! ========== [opencv_test10()]

// ! ========== [opencv_test11()]
int display_caption(const char *caption , const char *window_name , const cv::Mat *src , cv::Mat **dst);
int display_dst( const char *window_name , const cv::Mat *dst);

void opencv_test11(const int *argc , char **argv)
{
     if (*argc != 2){
         std::cout << "usage : <program> <image>" << std::endl;
         return ;
     }

     int MAX_KERNEL_LENGTH = 31;
     const char *window_name = "Smoothing Demo";
     cv::Mat *src = new cv::Mat();
     cv::Mat *dst = new cv::Mat();

     *src = cv::imread(argv[1] , cv::IMREAD_COLOR);
     cv::namedWindow(window_name , cv::WINDOW_AUTOSIZE);

     if (display_caption("Original Image" , window_name , src , &dst) != 0)
         return ;

     *dst = src->clone();
     if (display_dst( window_name , dst) != 0)
         return ;

     // 使用 均值平滑
     if (display_caption("Homogeneous Blur" , window_name , src ,&dst) != 0)
         return ;

     //! [高期平滑]
     for (int i = 1; i < MAX_KERNEL_LENGTH; i = i +2){
         cv::GaussianBlur(*src , *dst , cv::Size(i , i) , 0 , 0);
         if (display_dst( window_name , dst) != 0)
             return ;
     }
     //! [高期平滑]

     // 应用中值平滑
     if (display_caption("Median Blur" , window_name , src , &dst) != 0)
         return;

     //! [中值平滑]
     for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2){
         cv::medianBlur(*src , *dst , i);
         if (display_dst( window_name , dst) != 0)
             return;
     }
     //! [中值平滑]

     // 应用双边过滤
     if (display_caption("Bilateral Blur" , window_name , src , &dst) != 0)
         return;

     //! [双边过滤]
     for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2){
         cv::bilateralFilter(*src , *dst , i , i * 2 , i / 2);
         if (display_dst( window_name , dst) != 0)
             return ;
     }
     //! [双边过滤]

     display_caption("End: Press a key!" , window_name , src , &dst);
     cv::waitKey(0);

     std::cout << "程序正常退出 ， 记得释放指针!" << std::endl;
     deletePtr(src , "src" , __func__ , __LINE__);
     deletePtr(dst , "dst" , __func__ , __LINE__);
}

int display_caption(const char *caption , const char *window_name ,  const cv::Mat *src , cv::Mat **dst)
{
    cv::Mat *tmp = *dst;
    *tmp = cv::Mat::zeros(src->size() , src->type());
    cv::putText(*tmp , caption ,
                cv::Point(src->cols / 4 , src->rows / 2) ,
            cv::FONT_HERSHEY_COMPLEX , 1 , cv::Scalar(0 , 0 , 255));
    cv::imshow(window_name , *tmp);
    int c = cv::waitKey(0);
    return 0;
}

int display_dst( const char *window_name , const cv::Mat *dst)
{
    cv::imshow( window_name , *dst);
    cv::waitKey(0);
    return 0;
}

// ! ========== [opencv_test11()]

int main(int argc , char *argv[])
{
    //opencv_test1();  // 将cv::Mat 写入文件
    //opencv_test2(&argc , argv);  //彩色图片转灰色图片
    //opencv_test3();  //点的操作 point
    //opencv_test4(&argc , argv);   //扫描图片 , 最快的扫描图片方法是cv::LUT()函数 ，在教程中的介绍
    //opencv_test5(&argc , argv);  //矩阵上的掩码操作
    //opencv_test6(&argc , argv);  //迭代像素点
    //opencv_test7(&argc , argv);   // 两张图片组合显示
    //opencv_test8(&argc , argv);    // 调试图像对比度和亮度
    //opencv_test9(); // 画图，圆 ，椭圆 ，线 ， 多边形
    //opencv_test10(); // 画图训练
    //opencv_test11(&argc , argv); //图像模糊化

    auto glambda = [](auto a , auto&& b) {
        return a < b;
    };

    bool b = glambda(3 , 2.3);


    return 0;
}
