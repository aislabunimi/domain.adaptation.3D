# CMake generated Testfile for 
# Source directory: /home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/minkindr/minkindr
# Build directory: /home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/minkindr
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_minkindr_gtest_minkindr_tests "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/minkindr/catkin_generated/env_cached.sh" "/home/michele/miniconda3/envs/habitat/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/minkindr/test_results/minkindr/gtest-minkindr_tests.xml" "--return-code" "/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/merged/lib/minkindr/minkindr_tests --gtest_output=xml:/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/build/minkindr/test_results/minkindr/gtest-minkindr_tests.xml")
set_tests_properties(_ctest_minkindr_gtest_minkindr_tests PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;98;catkin_run_tests_target;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;37;_catkin_add_google_test;/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/minkindr/minkindr/CMakeLists.txt;9;catkin_add_gtest;/home/michele/Desktop/Colombo/tesi.triennale.colombo/catkin_ws/src/dipendenze_kimera/minkindr/minkindr/CMakeLists.txt;0;")
subdirs("gtest")
