import RPi.GPIO as GPIO
import time

# Use BCM GPIO numbering
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pins for output
led_pin1 = 17
led_pin2 = 27
buzzer_pin= 19

GPIO.setup(led_pin1, GPIO.OUT)
GPIO.setup(led_pin2, GPIO.OUT)

def turn_buzzer_on():
    GPIO.output(buzzer_pin, GPIO.HIGH)

def turn_buzzer_off():
    GPIO.output(buzzer_pin, GPIO.LOW)

def turn_led_on():
    GPIO.output(led_pin1, GPIO.HIGH)
    GPIO.output(led_pin2, GPIO.HIGH)

def turn_led_off():
    GPIO.output(led_pin1, GPIO.LOW)
    GPIO.output(led_pin2, GPIO.LOW)

def startup():
    turn_led_on()
    turn_buzzer_on()
    time.sleep(2)
    turn_buzzer_off()
    time.sleep(1)
    turn_buzzer_on()
    time.sleep(1)
    turn_buzzer_off()
    time.sleep(1)
    turn_buzzer_on()
    time.sleep(1)
    turn_buzzer_off()
    time.sleep(1)
    turn_led_off

try:
    startup()

finally:
    GPIO.cleanup()
         
