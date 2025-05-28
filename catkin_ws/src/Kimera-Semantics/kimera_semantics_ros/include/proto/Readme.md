# Setting up protobuf to allow export
Run the following to compile the ProtoBuf message.
```
protoc -I=$(rospack find kimera_semantics_ros)/include/proto/ --cpp_out=$(rospack find kimera_semantics_ros)/include/proto/ $(rospack find kimera_semantics_ros)/include/proto/semantic_map.proto
```


```
protoc -I=$(rospack find kimera_semantics_ros)/include/proto/ --python_out=$(rospack find kimera_semantics_ros)/include/proto/ $(rospack find kimera_semantics_ros)/include/proto/semantic_map.proto

protoc -I=$(rospack find kimera_semantics_ros)/include/proto/ --python_out=$(rospack find kimera_interfacer)/scripts/ $(rospack find kimera_semantics_ros)/include/proto/semantic_map.proto
```