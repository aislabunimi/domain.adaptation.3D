# Install script for directory: /home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install" TYPE PROGRAM FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/_setup_util.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install" TYPE PROGRAM FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/env.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/setup.bash;/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install" TYPE FILE FILES
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/setup.bash"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/local_setup.bash"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/setup.sh;/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install" TYPE FILE FILES
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/setup.sh"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/local_setup.sh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/setup.zsh;/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install" TYPE FILE FILES
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/setup.zsh"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install" TYPE FILE FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/.rosinstall")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/voxblox_msgs/msg" TYPE FILE FILES
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/msg/Block.msg"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/msg/Layer.msg"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/msg/Mesh.msg"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/msg/MeshBlock.msg"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/msg/VoxelEvaluationDetails.msg"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/voxblox_msgs/srv" TYPE FILE FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/srv/FilePath.srv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/voxblox_msgs/cmake" TYPE FILE FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/voxblox_msgs-msg-paths.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/include/voxblox_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/share/roseus/ros/voxblox_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/share/common-lisp/ros/voxblox_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/share/gennodejs/ros/voxblox_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/michele/miniconda3/envs/habitat/bin/python3" -m compileall "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/lib/python3/dist-packages/voxblox_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/lib/python3/dist-packages/voxblox_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/voxblox_msgs.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/voxblox_msgs/cmake" TYPE FILE FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/voxblox_msgs-msg-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/voxblox_msgs/cmake" TYPE FILE FILES
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/voxblox_msgsConfig.cmake"
    "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/catkin_generated/installspace/voxblox_msgsConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/voxblox_msgs" TYPE FILE FILES "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/voxblox/voxblox_msgs/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/gtest/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/voxblox_msgs/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
