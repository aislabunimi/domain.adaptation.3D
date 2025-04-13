# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "disparity_image_proc;cv_bridge;gflags_catkin;glog_catkin;minkindr_conversions;pcl_conversions;pcl_ros;sensor_msgs;tf;depth_image_proc;image_geometry;kimera_semantics;voxblox;voxblox_ros;voxblox_msgs;voxblox_rviz_plugin".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lkimera_semantics_ros".split(';') if "-lkimera_semantics_ros" != "" else []
PROJECT_NAME = "kimera_semantics_ros"
PROJECT_SPACE_DIR = "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/install"
PROJECT_VERSION = "1.0.0"
