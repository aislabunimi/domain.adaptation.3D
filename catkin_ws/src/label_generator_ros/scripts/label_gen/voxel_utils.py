import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32
from .proto.semantic_map_pb2 import SemanticMapProto
from .helper import get_grid_index_from_point, get_a_b_c_from_linear

EPS = 1e-4

def parse_protobuf_file(file_handle, msg):
    buf = file_handle.read()
    n = 0
    while n < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, n)
        n = new_pos
        msg_buf = buf[n:n + msg_len]
        n += msg_len
        msg.ParseFromString(msg_buf)
    return msg

def get_semantic_map(path):
    msg = SemanticMapProto()
    with open(path, "rb") as f:
        msg = parse_protobuf_file(f, msg)
    return msg

def parse_proto_to_numpy_array(map_msg):
    voxels_per_side = map_msg.semantic_blocks[0].voxels_per_side
    voxel_size = map_msg.semantic_blocks[0].voxel_size

    origins = np.array([
        [block.origin.x, block.origin.y, block.origin.z]
        for block in map_msg.semantic_blocks
    ])

    mi = np.min(origins, axis=0)
    ma = np.max(origins, axis=0) + (voxel_size * voxels_per_side)

    grid_extent = np.floor((ma - mi + EPS) / (voxel_size * voxels_per_side)).astype(np.uint32)
    full_shape = tuple(grid_extent * voxels_per_side)

    voxels = np.zeros((*full_shape, 41), dtype=np.float32)

    for idx, block in enumerate(map_msg.semantic_blocks):
        block_origin = origins[idx]
        block_idx = get_grid_index_from_point(block_origin - mi, 1.0 / (voxel_size * voxels_per_side))
        block_idx *= voxels_per_side

        for sem_voxel in block.semantic_voxels:
            abc = get_a_b_c_from_linear(sem_voxel.linear_index, voxels_per_side)
            voxel_idx = block_idx + abc
            voxels[tuple(voxel_idx)] = sem_voxel.semantic_labels

    return voxels, mi
