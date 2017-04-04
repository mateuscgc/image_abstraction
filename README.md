# image_abstraction
An implementation of the "Flow-Based Image Abstraction" article methods by Henry Kang et al.

Article method scheme:
[Article method scheme](docs/scheme.png?raw=true)

## OpenCV Installation (by [@AdeilsonSilva](https://github.com/AdeilsonSilva/))
#### INSTALL REQUIRED PACKAGES #####
```
$ sudo apt-get install build-essential cmake pkg-config libgtk2.0-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```
```
$ sudo apt-get install libpng12-0 libpng12-dev libpng++-dev libpng3 libpnglite-dev zlib1g-dbg zlib1g zlib1g-dev pngtools libjasper-dev libjasper-runtime libjasper1 libjpeg8 libjpeg8-dbg libjpeg62 libjpeg62-dev libjpeg-progs libtiff-tools libavcodec-dev libavformat-dev libswscale-dev openexr libopenexr6 libopenexr-dev
```
```
$ sudo apt-get install libgstreamer0.10-0-dbg libgstreamer0.10-0 libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libxine1-ffmpeg libxine-dev libxine1-bin libunicap2 libunicap2-dev libdc1394-22-dev libdc1394-22 libdc1394-utils libv4l-0 libv4l-dev
```

#### INSTALL OPENCV ####
```
$ git clone https://github.com/Itseez/opencv.git
```
```
$ git clone https://github.com/Itseez/opencv_contrib.git
```
```
$ cd opencv/ 
```
```
$ mkdir build
```
```
$ cd build
```
```
$ sudo cmake cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_DOCS=ON -D BUILD_EXAMPLES=ON -D WITH_GSTREAMER=ON -D WITH_IPP=OFF -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D INSTALL_C_EXAMPLES=ON -D CMAKE_INSTALL_PREFIX=/usr/local ..
```
```
$ sudo make -j3
```
```
$ sudo make install
```
*** The flag OPENCV_EXTRA_MODULES_PATH should always point to opencv_contrib/modules.

*** The flag CMAKE_INSTALL_PREFIX=/usr/local must stay like this so you won't need to change the following steps.

#### CONFIGURATION ####
```
$ sudo gedit /etc/ld.so.conf.d/opencv.conf
```
*** Write '/usr/local/lib' to the file above
```
$ sudo ldconfig -v
```
```
$ sudo gedit /etc/bash.bashrc
```
*** Write the two following lines to the file above:
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```
