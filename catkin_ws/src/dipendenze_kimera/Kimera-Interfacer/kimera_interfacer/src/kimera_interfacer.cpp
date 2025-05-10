#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>

#include <glog/logging.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <voxblox_ros/esdf_server.h>
#include <voxblox_ros/tsdf_server.h>
#include "kimera_interfacer/SyncSemantic.h"
#include "kimera_semantics_ros/depth_map_to_pointcloud.h"
#include "kimera_semantics_ros/semantic_tsdf_server.h"

#include <tf2_ros/transform_listener.h>
#include <minkindr_conversions/kindr_tf.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using namespace cv;

// Global variables
bool stop_generation = false;
std::unique_ptr<kimera::SemanticTsdfServer> tsdf_server;
sensor_msgs::Image::Ptr depth_ptr;
sensor_msgs::Image::Ptr img_ptr;
sensor_msgs::Image::Ptr seg_ptr;
sensor_msgs::CameraInfo::Ptr camera_ptr;
kimera::PointCloudFromDepth pcl_from_depth;
ros::Publisher pcl_pub;
int last_frame_integrated = 0;

// Params
std::string semantic_filename, mesh_filename, tsdf_esdf_filename;
std::string base_link_frame_id_, world_frame_id_;
ros::NodeHandle* nh_ptr;
ros::NodeHandle* nh_private_ptr;
tf2_ros::Buffer* tf_buffer_ptr;

void StopCallback(const std_msgs::Bool::ConstPtr &msg)
{
  if (msg->data)
  {
    stop_generation = true;
    ROS_INFO("Received stop signal. Saving map and resetting...");
  }
}

void SyncSemanticCallback(const kimera_interfacer::SyncSemantic::ConstPtr &msg, sensor_msgs::Image::Ptr &img, sensor_msgs::Image::Ptr &depth, sensor_msgs::Image::Ptr &seg)
{
  *img = msg->image;
  *depth = msg->depth;
  *seg = msg->sem;
}

void UpdateCameraCallback(const sensor_msgs::CameraInfo::ConstPtr &msg, sensor_msgs::CameraInfo::Ptr &pointer)
{
  *pointer = *msg;
}

void saveAndResetMap()
{
  // Save semantic voxel layer and mesh
  tsdf_server->serializeVoxelLayer(semantic_filename);
  tsdf_server->generateMesh();
  tsdf_server->saveMap(tsdf_esdf_filename);

  // Generate and save ESDF -- PROBABLY NOT NEEDED AND GENERATES ERRORS FOR SUBSCRIBERS not blocking 
  static constexpr bool kGenerateEsdf = true;
  voxblox::EsdfServer esdf_server(*nh_ptr, *nh_private_ptr); 
  esdf_server.loadMap(tsdf_esdf_filename);
  esdf_server.disableIncrementalUpdate();

  if (kGenerateEsdf ||
      esdf_server.getEsdfMapPtr()->getEsdfLayerPtr()->getNumberOfAllocatedBlocks() == 0)
  {
    static constexpr bool kFullEuclideanDistance = true;
    esdf_server.updateEsdfBatch(kFullEuclideanDistance);
  }

  esdf_server.saveMap(tsdf_esdf_filename);
  ROS_INFO("Map saved. Resetting state...");

  // Reset
  tsdf_server->clear();
  depth_ptr.reset(new sensor_msgs::Image);
  img_ptr.reset(new sensor_msgs::Image);
  seg_ptr.reset(new sensor_msgs::Image);
  camera_ptr.reset(new sensor_msgs::CameraInfo);
  last_frame_integrated = 0;
  stop_generation = false;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kimera_interfacer");
  google::InitGoogleLogging(argv[0]);
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  nh_ptr = &nh;
  nh_private_ptr = &nh_private;

  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  // Message pointers
  depth_ptr.reset(new sensor_msgs::Image);
  img_ptr.reset(new sensor_msgs::Image);
  seg_ptr.reset(new sensor_msgs::Image);
  camera_ptr.reset(new sensor_msgs::CameraInfo);

  // Params
  std::string sync_topic;
  nh_private.getParam("sync_topic", sync_topic);
  CHECK(nh_private.getParam("base_link_frame", base_link_frame_id_));
  CHECK(nh_private.getParam("world_frame", world_frame_id_));
  CHECK(nh_private.getParam("semantic_filename", semantic_filename));
  CHECK(nh_private.getParam("mesh_filename", mesh_filename));
  CHECK(nh_private.getParam("tsdf_filename", tsdf_esdf_filename));

  bool verbose;
  nh_private.param("verbose", verbose, false);

  // Subscribers
  ros::Subscriber sync_sub = nh.subscribe<kimera_interfacer::SyncSemantic>(sync_topic, 10,
    boost::bind(&SyncSemanticCallback, _1, img_ptr, depth_ptr, seg_ptr));
  ros::Subscriber stop_sub = nh.subscribe<std_msgs::Bool>("/kimera/end_generation_map", 10, StopCallback);
  ros::Subscriber camera_sub = nh.subscribe<sensor_msgs::CameraInfo>("depth_camera_info", 10,
    boost::bind(&UpdateCameraCallback, _1, camera_ptr));

  // Publisher
  pcl_pub = nh.advertise<kimera::PointCloud>("pcl", 10, true);

  // TF
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  tf_buffer_ptr = &tfBuffer;

  // TSDF server
  tsdf_server = kimera::make_unique<kimera::SemanticTsdfServer>(nh, nh_private ,verbose); // ADD VERBOSE IF NECESSARY

  ros::Rate r(50);
  ros::AsyncSpinner spinner(2);
  spinner.start();

  while (ros::ok())
  {
    bool new_frame = depth_ptr->header.seq != last_frame_integrated;

    if (stop_generation)
    {
      saveAndResetMap();
      continue;
    }

    if (new_frame)
    {
      last_frame_integrated = depth_ptr->header.seq;
      kimera::PointCloud::Ptr pcl = pcl_from_depth.imageCb(depth_ptr, seg_ptr, camera_ptr);

      try
      {
        geometry_msgs::TransformStamped geo = tfBuffer.lookupTransform(base_link_frame_id_, world_frame_id_, depth_ptr->header.stamp);
        tf::StampedTransform tf_transform;
        tf::transformStampedMsgToTF(geo, tf_transform);

        voxblox::Transformation T_G_B;
        tf::transformTFToKindr(tf_transform, &T_G_B);

        voxblox::Transformation T_B_C = T_G_B.inverse();
        tsdf_server->processPointCloudMessageAndInsert(pcl, T_B_C, false);
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
  return EXIT_SUCCESS;
}
