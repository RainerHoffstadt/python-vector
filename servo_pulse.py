from __future__ import division
import time
import Adafruit_PCA9685
#Aktiviert den Spannungsumsetzer vor dem PCA9685
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(7,GPIO.OUT, initial=GPIO.HIGH)

# Initalisierung mit alternativer Adresse
pwm = Adafruit_PCA9685.PCA9685(address=0x41)
pwm.set_pwm_freq(50)

def set_servo_pulse(channel, pulse):
    pulse_length = 1000000
    pulse_length /= 50
    print('{0}us per period'.format(pulse_length))
    pulse_length /= 4096
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    print(pulse_length)
    pulse /= pulse_length
    print(pulse)
    pulse = round(pulse)
    print(pulse)
    pulse = int(pulse)
    print (pulse)
    pwm.set_pwm(channel, 0, pulse)
    print("is run")
