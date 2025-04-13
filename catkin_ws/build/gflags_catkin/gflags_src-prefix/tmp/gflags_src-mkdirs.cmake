# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src")
  file(MAKE_DIRECTORY "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src")
endif()
file(MAKE_DIRECTORY
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src-build"
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix"
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/tmp"
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src-stamp"
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src"
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/gflags_catkin/gflags_src-prefix/src/gflags_src-stamp${cfgdir}") # cfgdir has leading slash
endif()
