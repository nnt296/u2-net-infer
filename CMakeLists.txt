cmake_minimum_required(VERSION 3.13)

project(salient-obj-detector)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV 4.5.2 EXACT PATHS /opt/minus/libs/opencv-4.5.2 NO_DEFAULT_PATH)

find_package(Torch 1.8.1 EXACT PATHS /opt/minus/libs/libtorch-gpu-1.8.1 NO_DEFAULT_PATH)

SET(SRC_FILES ${SRC_FILES}
        src/salient_detector.cpp
        src/common.cpp)

SET(HEADER_FILES ${HEADER_FILES}
        src/salient_detector.h
        src/common.h)

add_executable(
        ${PROJECT_NAME}
        main.cpp
        ${SRC_FILES}
        ${HEADER_FILES}
)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        boost_system
        boost_filesystem
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)