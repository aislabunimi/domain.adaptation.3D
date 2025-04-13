# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "kimera_interfacer: 2 messages, 0 services")

set(MSG_I_FLAGS "-Ikimera_interfacer:/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(kimera_interfacer_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" NAME_WE)
add_custom_target(_kimera_interfacer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "kimera_interfacer" "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" "sensor_msgs/Image:std_msgs/Header"
)

get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" NAME_WE)
add_custom_target(_kimera_interfacer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "kimera_interfacer" "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" "sensor_msgs/Image:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/kimera_interfacer
)
_generate_msg_cpp(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/kimera_interfacer
)

### Generating Services

### Generating Module File
_generate_module_cpp(kimera_interfacer
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/kimera_interfacer
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(kimera_interfacer_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(kimera_interfacer_generate_messages kimera_interfacer_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_cpp _kimera_interfacer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_cpp _kimera_interfacer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(kimera_interfacer_gencpp)
add_dependencies(kimera_interfacer_gencpp kimera_interfacer_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS kimera_interfacer_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/kimera_interfacer
)
_generate_msg_eus(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/kimera_interfacer
)

### Generating Services

### Generating Module File
_generate_module_eus(kimera_interfacer
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/kimera_interfacer
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(kimera_interfacer_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(kimera_interfacer_generate_messages kimera_interfacer_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_eus _kimera_interfacer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_eus _kimera_interfacer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(kimera_interfacer_geneus)
add_dependencies(kimera_interfacer_geneus kimera_interfacer_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS kimera_interfacer_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/kimera_interfacer
)
_generate_msg_lisp(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/kimera_interfacer
)

### Generating Services

### Generating Module File
_generate_module_lisp(kimera_interfacer
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/kimera_interfacer
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(kimera_interfacer_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(kimera_interfacer_generate_messages kimera_interfacer_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_lisp _kimera_interfacer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_lisp _kimera_interfacer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(kimera_interfacer_genlisp)
add_dependencies(kimera_interfacer_genlisp kimera_interfacer_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS kimera_interfacer_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/kimera_interfacer
)
_generate_msg_nodejs(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/kimera_interfacer
)

### Generating Services

### Generating Module File
_generate_module_nodejs(kimera_interfacer
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/kimera_interfacer
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(kimera_interfacer_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(kimera_interfacer_generate_messages kimera_interfacer_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_nodejs _kimera_interfacer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_nodejs _kimera_interfacer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(kimera_interfacer_gennodejs)
add_dependencies(kimera_interfacer_gennodejs kimera_interfacer_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS kimera_interfacer_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/kimera_interfacer
)
_generate_msg_py(kimera_interfacer
  "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/kimera_interfacer
)

### Generating Services

### Generating Module File
_generate_module_py(kimera_interfacer
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/kimera_interfacer
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(kimera_interfacer_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(kimera_interfacer_generate_messages kimera_interfacer_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemantic.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_py _kimera_interfacer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/Kimera-Interfacer/kimera_interfacer/msg/SyncSemanticRaw.msg" NAME_WE)
add_dependencies(kimera_interfacer_generate_messages_py _kimera_interfacer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(kimera_interfacer_genpy)
add_dependencies(kimera_interfacer_genpy kimera_interfacer_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS kimera_interfacer_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/kimera_interfacer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/kimera_interfacer
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(kimera_interfacer_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/kimera_interfacer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/kimera_interfacer
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(kimera_interfacer_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/kimera_interfacer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/kimera_interfacer
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(kimera_interfacer_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/kimera_interfacer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/kimera_interfacer
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(kimera_interfacer_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/kimera_interfacer)
  install(CODE "execute_process(COMMAND \"/home/michele/miniconda3/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/kimera_interfacer\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/kimera_interfacer
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(kimera_interfacer_generate_messages_py sensor_msgs_generate_messages_py)
endif()
