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
#include <mutex>
#include <minkindr_conversions/kindr_tf.h>

#include <std_msgs/Float32.h>
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
std::mutex frame_mutex;
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
  std::lock_guard<std::mutex> lock(frame_mutex);
 
  // Save semantic voxel layer and mesh
  tsdf_server->serializeVoxelLayer(semantic_filename);
  ros::Duration(1.0).sleep();
  tsdf_server->generateMesh();
  ROS_INFO("Map Generated ...");
  ros::Duration(1.0).sleep();
  tsdf_server->saveMap(tsdf_esdf_filename);
  ROS_INFO("Tsdf map saved ...");
  
  // Aggiungi una pausa di 1 secondo tra il salvataggio e il reset
  ros::Duration(2.0).sleep();  // 1 secondo di pausa

  // Generate and save ESDF -- PROBABLY NOT NEEDED AND GENERATES ERRORS FOR SUBSCRIBERS not blocking 
  static constexpr bool kGenerateEsdf = true;
  voxblox::EsdfServer esdf_server(*nh_ptr, *nh_private_ptr); 
  esdf_server.loadMap(tsdf_esdf_filename);
  esdf_server.disableIncrementalUpdate();

  if (kGenerateEsdf || esdf_server.getEsdfMapPtr()->getEsdfLayerPtr()->getNumberOfAllocatedBlocks() == 0)
  {
    static constexpr bool kFullEuclideanDistance = true;
    esdf_server.updateEsdfBatch(kFullEuclideanDistance);
  }

  esdf_server.saveMap(tsdf_esdf_filename);
  ROS_INFO("Esdf map saved. Resetting state ...");
  // Aggiungi una pausa prima di chiamare il clear
  ros::Duration(1.0).sleep();  // 1 secondo di pausa

  // Reset
  
  tsdf_server->clear();
  depth_ptr.reset(new sensor_msgs::Image);
  img_ptr.reset(new sensor_msgs::Image);
  seg_ptr.reset(new sensor_msgs::Image);
  camera_ptr.reset(new sensor_msgs::CameraInfo);
  last_frame_integrated = 0;
  stop_generation = false;
  ROS_INFO("State resetted!");
  return;
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
  ros::Subscriber sync_sub = nh.subscribe<kimera_interfacer::SyncSemantic>(sync_topic, 1,
    boost::bind(&SyncSemanticCallback, _1, img_ptr, depth_ptr, seg_ptr));
  ros::Subscriber stop_sub = nh.subscribe<std_msgs::Bool>("/kimera/end_generation_map", 1, StopCallback);
  ros::Subscriber camera_sub = nh.subscribe<sensor_msgs::CameraInfo>("depth_camera_info", 1,
    boost::bind(&UpdateCameraCallback, _1, camera_ptr));

  // Publisher
  pcl_pub = nh.advertise<kimera::PointCloud>("pcl", 10, true);
  ros::Publisher integration_time_pub = nh.advertise<std_msgs::Float32>("/kimera/integration_duration", 10);


  // TF
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  tf_buffer_ptr = &tfBuffer;

  // TSDF server
  tsdf_server = kimera::make_unique<kimera::SemanticTsdfServer>(nh, nh_private ,verbose); // ADD VERBOSE IF NECESSARY

  ros::Rate r(50);
  

  while (ros::ok())
  {
    ros::spinOnce(); 
    bool new_frame = depth_ptr->header.seq != last_frame_integrated;

    if (stop_generation)
    {
      saveAndResetMap();
      ros::Duration(1.0).sleep();
      continue;
    }

    if (new_frame)
    {
      std::lock_guard<std::mutex> lock(frame_mutex);
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

        // Timer start
        auto start_time = std::chrono::steady_clock::now();

        tsdf_server->processPointCloudMessageAndInsert(pcl, T_B_C, false);

        // Timer end
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = end_time - start_time;

        std_msgs::Float32 msg;
        msg.data = elapsed.count();  // seconds
        integration_time_pub.publish(msg);

        pcl_pub.publish(pcl);
      }
      catch (tf2::TransformException &ex)
      {
        LOG(ERROR) << "TF lookup failed: " << ex.what();
      }
    }

    r.sleep();
  }

  return EXIT_SUCCESS;
}
