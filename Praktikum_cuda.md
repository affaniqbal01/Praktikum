Internship Report: Configuration and Utilization of CUDA with OpenCV
Abstract
This report provides a detailed account of the configuration and utilization of OpenCV with CUDA acceleration during an internship. The work involved setting up a development environment optimized for GPU-accelerated image and video processing, using both the core OpenCV library and extra modules from the opencv_contrib repository. The focus was on integrating CUDA to improve computational efficiency and leveraging OpenCV’s additional, experimental functionalities through extra modules. The configuration steps, system specifications, and the build process are outlined to demonstrate the results achieved during the internship.

1. Introduction
OpenCV (Open Source Computer Vision Library) is widely used for real-time computer vision applications. By integrating it with NVIDIA's CUDA (Compute Unified Device Architecture), performance in computationally intensive tasks can be significantly enhanced. The aim of this internship was to configure and build OpenCV with CUDA acceleration, along with the extra modules provided by the opencv_contrib repository. This report documents the process followed to configure and build the software environment, the results achieved, and the specific steps undertaken.

2. System Configuration
2.1 Hardware Specifications

The configuration was carried out on a system equipped with an NVIDIA GeForce RTX 4060 Ti GPU. The detailed specifications are as follows:

Driver Version: 535.183.01
CUDA Version: 12.2
GPU Memory: 16,380 MiB
Current Memory Usage: 624 MiB
Power Usage: 8W / 165W
The GPU was used to accelerate image and video processing tasks by utilizing CUDA, allowing for parallel computation and increased efficiency.

2.2 Software Configuration

The OpenCV library was compiled from source with custom configurations to ensure CUDA support. The opencv_contrib repository was also integrated to include experimental and additional modules not present in the core OpenCV library. The configuration process used CMake, as shown in the command below:

bash
Copy code
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_TBB=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_CUDA=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=7.5 \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/opencv/lib/python3.12/site-packages/ \
      -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv/bin/python \
      -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv/opencv_contrib-4.10.0/modules \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=8.9 ..
The CMake configuration was successfully executed, with the following messages indicating completion:

Configuration: Done (3.3s)
Generating: Done (1.0s)
Build files written to: /home/gles/Downloads/opencv/opencv-4.10.0/build
In addition to CUDA integration, the opencv_model_diagnostics target was also built. This target provides diagnostic tools that help analyze and evaluate models within OpenCV, ensuring optimal performance.

3. Repository for OpenCV’s Extra Modules
The opencv_contrib repository is intended for the development of experimental and contributed modules. These modules are not included in the official OpenCV release, as they may lack a stable API or are not yet thoroughly tested. Maintaining the binary compatibility and performance of OpenCV is critical, which is why unstable modules are separated from the core library.

Modules from opencv_contrib are developed separately and published in this repository. When a module matures and gains sufficient popularity, it is moved to the central OpenCV repository, where it receives production-quality support from the development team.

To build OpenCV with the extra modules, the following CMake command was used:

bash
Copy code
$ cd <opencv_build_directory>
$ cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
$ make -j5
As a result, OpenCV was built in the specified directory with all modules from the opencv_contrib repository. In cases where not all modules were required, specific modules were excluded using CMake options, as shown:

bash
Copy code
$ cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -DBUILD_opencv_legacy=OFF <opencv_source_directory>
4. Results
The integration of CUDA with OpenCV was completed successfully. The build process was verified by reviewing the output and ensuring the correct configurations were applied. The following key results were achieved:

OpenCV was built with CUDA acceleration enabled, ensuring efficient handling of image and video processing tasks.
The opencv_contrib repository was integrated, providing additional modules that extend OpenCV’s capabilities.
The opencv_model_diagnostics target was successfully built, enabling the use of diagnostic tools to analyze and evaluate models for performance optimization.
The system was optimized for CUDA’s parallel computing power, offering improved performance in computationally heavy tasks.
5. Discussion
The successful integration of CUDA with OpenCV significantly enhances performance, particularly for tasks involving large datasets or real-time processing. CUDA’s ability to parallelize computations enables faster processing times, which is crucial in applications like video analysis and machine learning. By incorporating extra modules from the opencv_contrib repository, additional experimental features were made available, further expanding the potential for application development.

The use of diagnostic tools from opencv_model_diagnostics allowed for the evaluation of models, enabling better debugging and performance tuning. As the extra modules from opencv_contrib continue to evolve, further improvements and more stable versions of these features are expected to be integrated into the main OpenCV repository.

6. Conclusion
The task of configuring OpenCV with CUDA support, along with the extra modules from the opencv_contrib repository, was completed successfully. The build process followed the intended steps, and all modules were integrated as required. The use of CUDA acceleration significantly enhances the computational efficiency of OpenCV applications, especially in image and video processing. The integration of experimental modules from opencv_contrib also expanded the functionality of OpenCV, providing a more versatile and powerful toolset for future development.#References
CUDA Programming Guide. (n.d.). Retrieved from Wikipedia( https://en.wikipedia.org/wiki/CUDA)
OpenCV Configuration Script. (n.d.). Retrieved from GitHub (https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
NVIDIA GeForce RTX 4060 Ti Specifications. (n.d.). Retrieved from YouTube (https://www.youtube.com/watch?v=-GY2gT2umpk&t=515s)
