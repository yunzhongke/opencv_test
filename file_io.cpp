#include <opencv2/core.hpp>
#include <iostream>
#include <string>

/*
 * 使用XML和YAML文件的输入和输出
 *
 */


class MyData
{
public:
    MyData();
    explicit MyData(int);
    void write(cv::FileStorage &fs) const;
    void read(const cv::FileNode &node);
public:
    int A;
    double X;
    std::string id;
};


MyData::MyData():A(0) , X(0) , id()
{

}

MyData::MyData(int) :A(97) ,X(CV_PI) , id("mydata1234")
{

}

void MyData::read(const cv::FileNode &node)
{
    A = (int)node["A"];
    X = (double)node["X"];
    id = (std::string)node["id"];
}

void MyData::write(cv::FileStorage &fs) const
{
    fs << "{" << "A" << A << "X" << X << "id" << id << "}";
}


static void write(cv::FileStorage &fs , const std::string& , const MyData &x)
{
    x.write(fs);
}

static void read(const cv::FileNode &node , MyData &x , const MyData &default_value = MyData())
{
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

static std::ostream& operator<<(std::ostream &out , const MyData &m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}" ;
    return out;
}


int main(int argc , char *argv[])
{
    if (argc != 2){
        std::cout << "usage : [program] [xml or yaml]" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    { //write
        cv::Mat *R = new cv::Mat();
        cv::Mat *T = new cv::Mat();
        MyData *m = new MyData(1);

        *R = cv::Mat_<uchar>::eye(3 , 3);
        *T = cv::Mat_<double>::zeros(3 , 1);

        cv::FileStorage fs(filename , cv::FileStorage::WRITE);
        fs << "iteratinoNr" << 100;
        fs << "strings" << "[" ;
        fs << "image1.jpg" << "Awesomeness" << "~/00000001.jpg";
        fs << "]";

        fs << "Maping";
        fs << "{" << "One" << 1;
        fs << "Two" << 2 << "}";

        fs << "R" << *R;
        fs << "T" << *T;

        fs << "MyData" << *m ;
        fs.release();
        std::cout << "write Done." << std::endl;

        delete R;
        delete T;
        delete m;
    }

    {// read
        std::cout << std::endl << "Reading: " << std::endl;
        cv::FileStorage fs;
        if (!fs.open(filename , cv::FileStorage::READ)){
            std::cerr << "open " << filename << "fail!" << std::endl;
            return -4;
        }

        int itNr;
        itNr = (int)fs["iteratinoNr"] ;
        std::cout << itNr ;
        if (!fs.isOpened()){
            std::cerr << "Fail to open" << filename << std::endl;
            return -2;
        }
        cv::FileNode n = fs["strings"];
        if (n.type() != cv::FileNode::SEQ){
            std::cerr << "strings is not a sequence! FAIL" << std::endl;
            return -3;
        }
        cv::FileNodeIterator it = n.begin();
        for (; it != n.end(); ++it){
            std::cout << (std::string)*it << std::endl;
        }

        n = fs["Maping"];
        std::cout << "Two " << (int)(n["Two"]) << ";";
        std::cout << "One " << (int)(n["One"]) << std::endl << std::endl;

        MyData m;
        cv::Mat R , T;

        fs["R"] >> R;
        fs["T"] >> T;
        fs["MyData"] >> m;

        std::cout << std::endl << "R = " << R << std::endl;
        std::cout << "T = " << T << std::endl << std::endl;
        std::cout << "MyData = " << std::endl << m << std::endl << std::endl;

        std::cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
        fs["NoneExisting"] >> m;
        std::cout << std::endl << "NonExisting = " << std::endl << m << std::endl;
    }
    std::cout << "Tip: Open up " << filename << " with a text editor to see the serialized data." << std::endl;

    return 0;
}
