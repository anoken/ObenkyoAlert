cmake_minimum_required(VERSION 3.5)
project(obenkyo_alert)

set(CMAKE_BUILD_TYPE Release)
set(TARGET_COMPILER "arm" CACHE STRING "Target compiler (arm or x64)")
set(Target obenkyo_alert_nanodet)

set(COMMON_FLAGS "-O3 -Wall")

if(TARGET_COMPILER STREQUAL "arm")
    set(ARM_FLAGS "-march=armv7-a -mfpu=neon -mfloat-abi=hard -ffast-math")
    set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
    set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_FLAGS} ${ARM_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${COMMON_FLAGS} ${ARM_FLAGS}")
    set(NCNN_INSTALL_DIR /opt/ncnn/install/arm/)
    set(OpenCV_DIR /opt/opencv/install/arm/lib/cmake/opencv4)
    set(ALSA_INCLUDE_DIRS /opt/alsa-lib-1.2.7.2/include)
    set(ALSA_LIBRARIES /opt/alsa-lib-1.2.7.2/lib/libasound.so)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${COMMON_FLAGS}")
    set(NCNN_INSTALL_DIR /opt/ncnn/install/x64/)
    set(OpenCV_DIR /opt/opencv/install/x64/lib/cmake/opencv4)
    find_package(ALSA REQUIRED)
endif()

find_package(OpenCV REQUIRED)

include_directories(${ALSA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${NCNN_INSTALL_DIR}/include/ncnn)

add_executable(${Target} ${Target}.cpp)

target_link_libraries(${Target}
    ${ALSA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${NCNN_INSTALL_DIR}/lib/libncnn.a
    pthread
    -fopenmp
)

