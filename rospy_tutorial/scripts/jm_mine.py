#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

if __name__ == '__main__':
	try:
		while not rospy.is_shutdown():
			rospy.init_node("jm_test",anonymous=True)
			rate = rospy.Rate(1)
			pb = rospy.Publisher("jm_test", String, queue_size=10)
			rospy.loginfo('hello world!')
			pb.publish("hello world!")
			rate.sleep()
			
	except rospy.ROSInterruptException:
		pass
			
