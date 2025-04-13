;; Auto-generated. Do not edit!


(when (boundp 'kimera_interfacer::SyncSemantic)
  (if (not (find-package "KIMERA_INTERFACER"))
    (make-package "KIMERA_INTERFACER"))
  (shadow 'SyncSemantic (find-package "KIMERA_INTERFACER")))
(unless (find-package "KIMERA_INTERFACER::SYNCSEMANTIC")
  (make-package "KIMERA_INTERFACER::SYNCSEMANTIC"))

(in-package "ROS")
;;//! \htmlinclude SyncSemantic.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass kimera_interfacer::SyncSemantic
  :super ros::object
  :slots (_depth _sem _image ))

(defmethod kimera_interfacer::SyncSemantic
  (:init
   (&key
    ((:depth __depth) (instance sensor_msgs::Image :init))
    ((:sem __sem) (instance sensor_msgs::Image :init))
    ((:image __image) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _depth __depth)
   (setq _sem __sem)
   (setq _image __image)
   self)
  (:depth
   (&rest __depth)
   (if (keywordp (car __depth))
       (send* _depth __depth)
     (progn
       (if __depth (setq _depth (car __depth)))
       _depth)))
  (:sem
   (&rest __sem)
   (if (keywordp (car __sem))
       (send* _sem __sem)
     (progn
       (if __sem (setq _sem (car __sem)))
       _sem)))
  (:image
   (&rest __image)
   (if (keywordp (car __image))
       (send* _image __image)
     (progn
       (if __image (setq _image (car __image)))
       _image)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _depth
    (send _depth :serialization-length)
    ;; sensor_msgs/Image _sem
    (send _sem :serialization-length)
    ;; sensor_msgs/Image _image
    (send _image :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _depth
       (send _depth :serialize s)
     ;; sensor_msgs/Image _sem
       (send _sem :serialize s)
     ;; sensor_msgs/Image _image
       (send _image :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _depth
     (send _depth :deserialize buf ptr-) (incf ptr- (send _depth :serialization-length))
   ;; sensor_msgs/Image _sem
     (send _sem :deserialize buf ptr-) (incf ptr- (send _sem :serialization-length))
   ;; sensor_msgs/Image _image
     (send _image :deserialize buf ptr-) (incf ptr- (send _image :serialization-length))
   ;;
   self)
  )

(setf (get kimera_interfacer::SyncSemantic :md5sum-) "06521f7c2a5c6fea82ae6a53446fde98")
(setf (get kimera_interfacer::SyncSemantic :datatype-) "kimera_interfacer/SyncSemantic")
(setf (get kimera_interfacer::SyncSemantic :definition-)
      "sensor_msgs/Image depth
sensor_msgs/Image sem
sensor_msgs/Image image
================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :kimera_interfacer/SyncSemantic "06521f7c2a5c6fea82ae6a53446fde98")


