cmake_minimum_required(VERSION 3.13)

project(test-so)

# For Analytic Service Headers to find Opencv Headers
find_package(OpenCV 4.5.2 EXACT PATHS /opt/minus/libs/opencv-4.5.2 NO_DEFAULT_PATH)

# Fix CMAKE configuration libtorch1.8.1 + co111 error failed to compute shorthash for libnvrtc.so
find_package(PythonInterp REQUIRED)

# For Analytic Service Headers to find Torch Headers
find_package(Torch 1.8.1 EXACT PATHS /opt/minus/libs/libtorch NO_DEFAULT_PATH)

set (INC_DIRS ${INC_DIRS} ${OpenCV_INCLUDE_DIRS})
set (LIBS ${LIBS} ${OpenCV_LIBS})
set (LIBS ${LIBS} ${TORCH_LIBRARIES})

# Add Analytic Service Headers and Shared Libraries
set (INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../libsegmentation/include/)
set (LIBS ${LIBS} ${CMAKE_CURRENT_SOURCE_DIR}/../libsegmentation/lib/libsegmentation.so)

add_executable(${PROJECT_NAME} main.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

target_include_directories(${PROJECT_NAME} PUBLIC ${INC_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS})
