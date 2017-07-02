TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += -I/home/yunzhongke/work/ws_opencv/include

LIBS += -L/home/yunzhongke/work/ws_opencv/lib \
        -lopencv_calib3d \
        -lopencv_core \
        -lopencv_cudaarithm \
        -lopencv_cudabgsegm \
        -lopencv_cudacodec \
        -lopencv_cudafeatures2d \
        -lopencv_cudafilters \
        -lopencv_cudaimgproc \
        -lopencv_cudalegacy \
        -lopencv_cudaobjdetect \
        -lopencv_cudaoptflow \
        -lopencv_cudastereo \
        -lopencv_cudawarping \
        -lopencv_cudev \
        -lopencv_features2d \
        -lopencv_flann \
        -lopencv_highgui \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_ml \
        -lopencv_objdetect \
        -lopencv_photo \
        -lopencv_shape \
        -lopencv_stitching \
        -lopencv_superres \
        -lopencv_videoio \
        -lopencv_video \
        -lopencv_videostab \
        -lopencv_viz

