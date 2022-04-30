import time
#import sys

from servo_pulse import set_servo_pulse


def close():
	set_servo_pulse(5, 0.7)
	time.sleep(10)
	set_servo_pulse(5, 1.5)

