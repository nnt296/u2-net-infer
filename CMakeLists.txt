cmake_minimum_required(VERSION 3.13)

project(salient-obj-detector)

set(CMAKE_BUILD_TYPE Debug)

# Fix CMAKE configuration libtorch1.8.1 + co111 error failed to compute shorthash for libnvrtc.so
find_package(PythonInterp REQUIRED)

find_package(OpenCV 4.5.2 EXACT PATHS /opt/minus/libs/opencv-4.5.2 NO_DEFAULT_PATH)

find_package(Torch 1.8.1 EXACT PATHS /opt/minus/libs/libtorch-gpu-1.8.1 NO_DEFAULT_PATH)

SET(SRC_FILES ${SRC_FILES}
        src/salient_detector.cpp
        src/sd_utils.cpp)

SET(HEADER_FILES ${HEADER_FILES}
        src/salient_detector.h
        src/sd_utils.h)

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