cmake_minimum_required(VERSION 3.13)

project(salient-obj-detector)

set(CMAKE_BUILD_TYPE Debug)

# Fix CMAKE configuration libtorch1.8.1 + co111 error failed to compute shorthash for libnvrtc.so
find_package(PythonInterp REQUIRED)

find_package(OpenCV 4.5.2 EXACT PATHS /opt/minus/libs/opencv-4.5.2 NO_DEFAULT_PATH)

find_package(Torch 1.8.1 EXACT PATHS /opt/minus/libs/libtorch NO_DEFAULT_PATH)

SET(SRC_FILES ${SRC_FILES}
        src/salient_detector.cpp
        src/sd_utils.cpp)

SET(HEADER_FILES ${HEADER_FILES}
        src/salient_detector.h
        src/sd_utils.h)

# BUILD EXECUTABLE FILE
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
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
# END BUILD EXECUTABLE FILE


# BUILD LIBS
SET(LIB_BUILD_NAME segmentation)
add_library(${LIB_BUILD_NAME} SHARED ${SRC_FILES} ${HEADER_FILES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${LIB_BUILD_NAME} PROPERTY CXX_STANDARD 14)

target_include_directories(
        ${LIB_BUILD_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        src/
)

target_link_libraries(
        ${LIB_BUILD_NAME} PUBLIC
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)

set(CMAKE_INSTALL_LIBDIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Install libs
install(TARGETS ${LIB_BUILD_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/libsegmentation/lib)

# Install header
install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/libsegmentation/include)

# END BUILD EXECUTABLE FILE
