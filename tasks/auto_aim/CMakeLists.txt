cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
set(RKNN_LIBRARY "/usr/lib/librknnrt.so")
set(RKNN_INCLUDE_DIR "../../include")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    fire.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    fire_yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    yolos/fire_yolov5.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

# 添加RKNN头文件包含路径
target_include_directories(auto_aim PRIVATE ${RKNN_INCLUDE_DIR})

# 链接RKNN库
target_link_libraries(auto_aim io openvino::runtime tinympcstatic ${RKNN_LIBRARY})