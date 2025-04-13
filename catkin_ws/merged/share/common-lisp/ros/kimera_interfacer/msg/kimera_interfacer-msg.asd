
(cl:in-package :asdf)

(defsystem "kimera_interfacer-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "SyncSemantic" :depends-on ("_package_SyncSemantic"))
    (:file "_package_SyncSemantic" :depends-on ("_package"))
    (:file "SyncSemanticRaw" :depends-on ("_package_SyncSemanticRaw"))
    (:file "_package_SyncSemanticRaw" :depends-on ("_package"))
  ))