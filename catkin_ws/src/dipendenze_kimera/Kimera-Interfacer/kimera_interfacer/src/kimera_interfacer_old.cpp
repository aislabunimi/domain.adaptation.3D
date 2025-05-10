#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <glog/logging.h>

#include <voxblox_ros/esdf_server.h>
#include <voxblox_ros/tsdf_server.h>
#include "kimera_interfacer/SyncSemantic.h"

#include "kimera_semantics_ros/depth_map_to_pointcloud.h"
#include "kimera_semantics_ros/rosbag_data_provider.h"
#include "kimera_semantics_ros/semantic_tsdf_server.h"

#include <iostream>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <filesystem>
#include <tf2_ros/transform_listener.h>
#include <minkindr_conversions/kindr_tf.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <std_msgs/Bool.h>

namespace fs = std::filesystem;
using namespace cv;
bool stop_generation = false;

void StopCallback(const std_msgs::Bool::ConstPtr &msg)
{
  if (msg->data)
  {
    stop_generation = true;
    ROS_INFO("Received stop signal. Stopping map generation.");
  }
}


bool fileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}


void SyncSemanticCallback(const kimera_interfacer::SyncSemantic::ConstPtr &msg, sensor_msgs::Image::Ptr &img, sensor_msgs::Image::Ptr &depth, sensor_msgs::Image::Ptr &seg)
{
  // performs a copy operation of the shared pointer to the buffer
  *img = msg->image;
  *depth = msg->depth;
  *seg = msg->sem;
}

void UpdateCameraCallback(const sensor_msgs::CameraInfo::ConstPtr &msg, sensor_msgs::CameraInfo::Ptr &pointer)
{
  *pointer = *msg;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kimera_interfacer");
  google::InitGoogleLogging(argv[0]);
  std::cout << "Started the Interfacer Node" << std::endl;
  VLOG(1) << "I'm printed when you run the program with --v=1 or higher";

  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  sensor_msgs::Image::Ptr depth_ptr(new sensor_msgs::Image);
  sensor_msgs::Image::Ptr img_ptr(new sensor_msgs::Image);
  sensor_msgs::Image::Ptr seg_ptr(new sensor_msgs::Image);

  std::string sync_topic;
  nh_private.getParam("sync_topic", sync_topic);
  std::cout << "SUBSCRIBING" << std::endl;
  ros::Subscriber sync_sub = nh.subscribe<kimera_interfacer::SyncSemantic>(sync_topic, 10,
                                                                           boost::bind(&SyncSemanticCallback, _1, img_ptr, depth_ptr, seg_ptr));
  ros::Subscriber stop_sub = nh.subscribe<std_msgs::Bool>("/kimera/end_generation_map", 10, StopCallback);
  std::string depth_cam_frame_id_;
  std::string base_link_frame_id_;
  std::string world_frame_id_;

  CHECK(nh_private.getParam("sensor_frame", depth_cam_frame_id_));
  CHECK(nh_private.getParam("base_link_frame", base_link_frame_id_));
  CHECK(nh_private.getParam("world_frame", world_frame_id_));

  std::unique_ptr<kimera::SemanticTsdfServer> tsdf_server;
  tsdf_server = kimera::make_unique<kimera::SemanticTsdfServer>(nh, nh_private);

  kimera::PointCloudFromDepth pcl_from_depth;

  // future rework use camera infor manager instead and use service
  sensor_msgs::CameraInfo::Ptr camera_ptr(new sensor_msgs::CameraInfo);
  ros::Subscriber camera_sub = nh.subscribe<sensor_msgs::CameraInfo>(
      "depth_camera_info", 10,
      boost::bind(&UpdateCameraCallback,
                  _1, camera_ptr));

  ros::Publisher pcl_pub;
  pcl_pub = nh.advertise<kimera::PointCloud>("pcl", 10, true);
  kimera::PointCloud::Ptr pcl;
  cv_bridge::CvImagePtr cv_ptr;
  ros::Rate r(50);

  // get transformation
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  int j = 0;
  int last_frame_integrated = 0;
  std::cout << "STARTING LOOP" << std::endl;
  ros::AsyncSpinner spinner(2);
  spinner.start();
  while (ros::ok())
  {

    bool res = depth_ptr->header.seq != last_frame_integrated;
    if (stop_generation)
    {
      break;
    }
    if (res)
    {

      j = 0;
      // std::cout << "Received new data and alread process " << j << "Frames"<< std::endl;
      last_frame_integrated = depth_ptr->header.seq;
      
      pcl = pcl_from_depth.imageCb(depth_ptr, seg_ptr, camera_ptr);
      //std::cout << "PCL generated height:" << pcl->height << " Width:" << pcl->width << pcl->fields[0].count << " DATA 0" << pcl->data[0] << std::endl;
      //  Feed semantic pointcloud to KS.
      try
      {
        voxblox::Transformation T_G_B;
        geometry_msgs::TransformStamped geo;
        geo = tfBuffer.lookupTransform(base_link_frame_id_,
                                       world_frame_id_,
                                       depth_ptr->header.stamp);
        tf::StampedTransform tf_transform;
        tf::transformStampedMsgToTF(geo, tf_transform);
        tf::transformTFToKindr(tf_transform, &T_G_B);

        voxblox::Transformation T_B_C;
        T_B_C = T_G_B.inverse();
        
        tsdf_server->processPointCloudMessageAndInsert(pcl, T_B_C, false);
        // pcl->header.frame_id = world_frame_id_;
        pcl_pub.publish(pcl);
      }
      catch (tf2::TransformException &ex)
      {
        LOG(ERROR) << "TF lookup failed: " << ex.what();
      }
    }

    r.sleep();
  }
  spinner.stop();

  std::string mesh_filename;
  std::string semantic_filename;
  std::string tsdf_esdf_filename;
  CHECK(nh_private.getParam("semantic_filename", semantic_filename));
  CHECK(nh_private.getParam("mesh_filename", mesh_filename));
  CHECK(nh_private.getParam("tsdf_filename", tsdf_esdf_filename));
  /*
  if (!fileExists(semantic_filename)) {
    ROS_WARN_STREAM("File not found: semantic_filename = " << semantic_filename);
    ROS_INFO("Exiting node gracefully. No map will be generated.");
    tsdf_server-> clear();
    return 0;
  }
  if (!fileExists(tsdf_esdf_filename)) {
    ROS_WARN_STREAM("File not found: tsdf_filename = " << tsdf_esdf_filename);
    ROS_INFO("Exiting node gracefully. No map will be generated.");
    tsdf_server-> clear();
    return 0;
  }
  */
  std::cout << "START GENERATING THE MESH" << std::endl;
  tsdf_server->serializeVoxelLayer(semantic_filename);
  std::cout << "FINISHED GENERATING THE MESH" << std::endl;

  tsdf_server->generateMesh();
  // Saves the TSDF layer
  tsdf_server->saveMap(tsdf_esdf_filename);

  // Generate the ESDF layer in batch.
  LOG(INFO) << "Start Batch ESDF generation.";
  static constexpr bool kGenerateEsdf = true;
  voxblox::EsdfServer esdf_server(nh, nh_private);
  esdf_server.loadMap(tsdf_esdf_filename);
  esdf_server.disableIncrementalUpdate();
  if (kGenerateEsdf ||
      esdf_server.getEsdfMapPtr()
              ->getEsdfLayerPtr()
              ->getNumberOfAllocatedBlocks() == 0)
  {
    static constexpr bool kFullEuclideanDistance = true;
    esdf_server.updateEsdfBatch(kFullEuclideanDistance);
  }
  // Save the ESDF layer (in the same file as the TSDF).
  esdf_server.saveMap(tsdf_esdf_filename);
  LOG(INFO) << "Finished Batch ESDF generation.";
  ros::spin();
  return EXIT_SUCCESS;
}
