// NOTE: Most code is derived from voxblox: github.com/ethz-asl/voxblox
// Copyright (c) 2016, ETHZ ASL
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of voxblox nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

/**
 * @file   semantic_tsdf_server.cpp
 * @brief  Semantic TSDF Server to interface with ROS
 * @author Antoni Rosinol
 */

#include "kimera_semantics_ros/semantic_tsdf_server.h"

#include <glog/logging.h>
#include <kimera_semantics/semantic_tsdf_integrator_factory.h>
#include <voxblox_ros/ros_params.h>

#include <fstream>
#include <iostream>
#include <string>

#include "kimera_semantics_ros/ros_params.h"
#include "proto/semantic_map.pb.h"

namespace kimera {

SemanticTsdfServer::SemanticTsdfServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private,
                                       bool verbose)
    : SemanticTsdfServer(nh,
                         nh_private,
                         vxb::getTsdfMapConfigFromRosParam(nh_private),
                         vxb::getTsdfIntegratorConfigFromRosParam(nh_private),
                         vxb::getMeshIntegratorConfigFromRosParam(nh_private),
                         verbose) {}

SemanticTsdfServer::SemanticTsdfServer(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& nh_private,
    const vxb::TsdfMap::Config& config,
    const vxb::TsdfIntegratorBase::Config& integrator_config,
    const vxb::MeshIntegratorConfig& mesh_config,
    bool verbose)
    : vxb::TsdfServer(nh, nh_private, config, integrator_config, mesh_config),
      semantic_config_(getSemanticTsdfIntegratorConfigFromRosParam(nh_private)),
      semantic_layer_(nullptr) {
  // 💡 This sets the inherited TsdfServer's verbosity flag.
  verbose_ = verbose;

  semantic_layer_.reset(new vxb::Layer<SemanticVoxel>(
      config.tsdf_voxel_size, config.tsdf_voxels_per_side));

  tsdf_integrator_ = SemanticTsdfIntegratorFactory::create(
      getSemanticTsdfIntegratorTypeFromRosParam(nh_private),
      integrator_config,
      semantic_config_,
      tsdf_map_->getTsdfLayerPtr(),
      semantic_layer_.get());

  CHECK(tsdf_integrator_);
}

void SemanticTsdfServer::serializeVoxelLayer(const std::string& file_path) {
  vxb::BlockIndex index;
  vxb::BlockIndexList blocks;
  semantic_layer_->getAllAllocatedBlocks(&blocks);

  // Create an empty object
  std::fstream outfile;
  outfile.open(file_path, std::fstream::out);

  if (outfile.is_open()) {
    SemanticMapProto semantic_map_proto;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
      vxb::BlockIndex semantic_block_idx;
      semantic_block_idx = *it;
      vxb::Block<SemanticVoxel>& semantic_block =
          semantic_layer_->getBlockByIndex(semantic_block_idx);
      vxb::GlobalIndex global_voxel_idx;

      size_t num_voxels = semantic_block.num_voxels();
      size_t voxel_per_side = semantic_block.voxels_per_side();
      auto voxel_size = semantic_block.voxel_size();
      // TODO: Jonas Frey make use of serialization of Protobuf using streams
      auto semantic_block_proto = semantic_map_proto.add_semantic_blocks();
      semantic_block_proto->set_voxel_size(voxel_size);
      semantic_block_proto->set_voxels_per_side(voxel_per_side);
      auto origin_proto = semantic_block_proto->mutable_origin();
      origin_proto->set_x(semantic_block.origin().x());
      origin_proto->set_y(semantic_block.origin().y());
      origin_proto->set_z(semantic_block.origin().z());
      for (size_t i = 0; i < num_voxels; i++) {
        SemanticVoxel& sem_voxel = semantic_block.getVoxelByLinearIndex(i);
        uint8_t label = sem_voxel.semantic_label;
        if (label != 0) {
          auto semantic_voxel_proto =
              semantic_block_proto->add_semantic_voxels();
          auto color_proto = semantic_voxel_proto->mutable_color();
          color_proto->set_r(sem_voxel.color.r);
          color_proto->set_g(sem_voxel.color.g);
          color_proto->set_b(sem_voxel.color.b);
          semantic_voxel_proto->set_linear_index(i);
          auto A = sem_voxel.semantic_priors;
          for (int i = 0; i < A.rows(); ++i) {
            auto value = A.row(i);
            semantic_voxel_proto->add_semantic_labels(float(value[i, 0]));
          }
        }
      }
    }
    auto res =
        voxblox::utils::writeProtoMsgToStream(semantic_map_proto, &outfile);
    outfile.close();
    std::cout << "Done writing to file for serialization" << std::endl;
  } else {
    std::cout << "Failed opening the file for serialization" << std::endl;
  }
}
}  // Namespace kimera