#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys, termios, tty

# Mapping frecce -> twist
key_mapping = {
    '\x1b[A': (1.0, 0.0, 0.0),   # freccia su -> avanti
    '\x1b[B': (-1.0, 0.0, 0.0),  # freccia giù -> indietro
    '\x1b[C': (0.0, 0.0, -1.0),  # freccia destra -> turn right
    '\x1b[D': (0.0, 0.0, 1.0),   # freccia sinistra -> turn left
}

def get_key():
    """Legge un tasto senza invio"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == '\x1b':  # escape sequence
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch1 + ch2 + ch3
        else:
            return ch1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    rospy.init_node('keyboard_cmd_vel')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rate = rospy.Rate(10)

    print("Controlla il robot con le frecce. CTRL-C per uscire.")

    while not rospy.is_shutdown():
        key = get_key()
        twist = Twist()

        if key in key_mapping:
            twist.linear.x = key_mapping[key][0]
            twist.linear.y = key_mapping[key][1]
            twist.angular.z = key_mapping[key][2]
        else:
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0

        pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
