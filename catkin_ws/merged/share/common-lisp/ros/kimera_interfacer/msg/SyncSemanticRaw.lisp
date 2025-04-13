; Auto-generated. Do not edit!


(cl:in-package kimera_interfacer-msg)


;//! \htmlinclude SyncSemanticRaw.msg.html

(cl:defclass <SyncSemanticRaw> (roslisp-msg-protocol:ros-message)
  ((depth
    :reader depth
    :initarg :depth
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (sem
    :reader sem
    :initarg :sem
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass SyncSemanticRaw (<SyncSemanticRaw>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SyncSemanticRaw>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SyncSemanticRaw)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name kimera_interfacer-msg:<SyncSemanticRaw> is deprecated: use kimera_interfacer-msg:SyncSemanticRaw instead.")))

(cl:ensure-generic-function 'depth-val :lambda-list '(m))
(cl:defmethod depth-val ((m <SyncSemanticRaw>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader kimera_interfacer-msg:depth-val is deprecated.  Use kimera_interfacer-msg:depth instead.")
  (depth m))

(cl:ensure-generic-function 'sem-val :lambda-list '(m))
(cl:defmethod sem-val ((m <SyncSemanticRaw>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader kimera_interfacer-msg:sem-val is deprecated.  Use kimera_interfacer-msg:sem instead.")
  (sem m))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <SyncSemanticRaw>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader kimera_interfacer-msg:image-val is deprecated.  Use kimera_interfacer-msg:image instead.")
  (image m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SyncSemanticRaw>) ostream)
  "Serializes a message object of type '<SyncSemanticRaw>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'depth) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'sem) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SyncSemanticRaw>) istream)
  "Deserializes a message object of type '<SyncSemanticRaw>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'depth) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'sem) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SyncSemanticRaw>)))
  "Returns string type for a message object of type '<SyncSemanticRaw>"
  "kimera_interfacer/SyncSemanticRaw")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SyncSemanticRaw)))
  "Returns string type for a message object of type 'SyncSemanticRaw"
  "kimera_interfacer/SyncSemanticRaw")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SyncSemanticRaw>)))
  "Returns md5sum for a message object of type '<SyncSemanticRaw>"
  "06521f7c2a5c6fea82ae6a53446fde98")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SyncSemanticRaw)))
  "Returns md5sum for a message object of type 'SyncSemanticRaw"
  "06521f7c2a5c6fea82ae6a53446fde98")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SyncSemanticRaw>)))
  "Returns full string definition for message of type '<SyncSemanticRaw>"
  (cl:format cl:nil "sensor_msgs/Image depth~%sensor_msgs/Image sem~%sensor_msgs/Image image~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SyncSemanticRaw)))
  "Returns full string definition for message of type 'SyncSemanticRaw"
  (cl:format cl:nil "sensor_msgs/Image depth~%sensor_msgs/Image sem~%sensor_msgs/Image image~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SyncSemanticRaw>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'depth))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'sem))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SyncSemanticRaw>))
  "Converts a ROS message object to a list"
  (cl:list 'SyncSemanticRaw
    (cl:cons ':depth (depth msg))
    (cl:cons ':sem (sem msg))
    (cl:cons ':image (image msg))
))
