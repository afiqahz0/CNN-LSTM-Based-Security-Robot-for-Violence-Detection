#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import socket
import threading
import statistics
import subprocess
import shlex
import os
import signal
import sys

GPIO.setmode(GPIO.BCM)

# -----------------------
# Camera stream config
# -----------------------
LAPTOP_IP = "192.168.137.1"    # <-- CHANGE if needed
UDP_PORT  = 5000

CAMERA_CMD = (
    "rpicam-vid "
    "--inline --listen "
    "--width 640 --height 360 --framerate 25 "
    "--codec h264 --profile high "
    "--flush "
    "-t 0 "
    f"-o udp://{LAPTOP_IP}:{UDP_PORT}"
)

camera_proc = None


def start_camera():
    """Start rpicam-vid as background process."""
    global camera_proc
    print("[CAM] Starting camera stream...")
    args = shlex.split(CAMERA_CMD)
    devnull = open(os.devnull, "wb")
    camera_proc = subprocess.Popen(
        args,
        stdout=devnull,
        stderr=devnull,
        preexec_fn=os.setsid
    )
    print(f"[CAM] Camera started (PID={camera_proc.pid})")


def stop_camera():
    """Stop the background camera process."""
    global camera_proc
    if camera_proc is None:
        return
    print("[CAM] Stopping camera...")
    try:
        os.killpg(os.getpgid(camera_proc.pid), signal.SIGTERM)
    except Exception:
        try:
            camera_proc.terminate()
        except:
            pass
    camera_proc = None
    print("[CAM] Camera stopped.")


# -----------------------
# Motor Driver Pins
# -----------------------
STBY = 22

MOTORS = {
    "FL": (23, 24, 18),
    "RL": (17, 27, 12),
    "FR": (5, 6, 13),
    "RR": (19, 26, 16)
}

for (in1, in2, pwm) in MOTORS.values():
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(pwm,  GPIO.OUT)

GPIO.setup(STBY, GPIO.OUT)
GPIO.output(STBY, GPIO.LOW)

pwms = {}
for name, (_, _, pwm) in MOTORS.items():
    p = GPIO.PWM(pwm, 1000)
    p.start(0)
    pwms[name] = p


# -----------------------
# Ultrasonic Sensor Pins
# -----------------------
SENSORS = {
    "Front": {"trig": 8,  "echo": 9},
    "Left":  {"trig": 25, "echo": 4},
    "Right": {"trig": 20, "echo": 21}
}

for s in SENSORS.values():
    GPIO.setup(s["trig"], GPIO.OUT)
    GPIO.setup(s["echo"], GPIO.IN)
    GPIO.output(s["trig"], GPIO.LOW)

time.sleep(0.2)

# -----------------------
# Parameters
# -----------------------
SAFE_DIST = 30
EMERGENCY_DIST = 10

FORWARD_SPEED = 30
TURN_SPEED = 30
BACK_SPEED = 30

BACK_TIME = 0.4
TURN_TIME = 0.45

LOOP_DELAY = 0.07
SAMPLE_COUNT = 3

# -----------------------
# Motor Helpers
# -----------------------
def enable_driver():  GPIO.output(STBY, GPIO.HIGH)
def disable_driver(): GPIO.output(STBY, GPIO.LOW)

def motor_forward(name, speed):
    in1, in2, _ = MOTORS[name]
    GPIO.output(in1, GPIO.HIGH); GPIO.output(in2, GPIO.LOW)
    pwms[name].ChangeDutyCycle(speed)

def motor_backward(name, speed):
    in1, in2, _ = MOTORS[name]
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.HIGH)
    pwms[name].ChangeDutyCycle(speed)

def motor_stop(name):
    in1, in2, _ = MOTORS[name]
    GPIO.output(in1, GPIO.LOW); GPIO.output(in2, GPIO.LOW)
    pwms[name].ChangeDutyCycle(0)

def all_forward(speed):
    for m in MOTORS:
        motor_forward(m, speed)

def all_backward(speed):
    for m in MOTORS:
        motor_backward(m, speed)

def all_stop():
    for m in MOTORS:
        motor_stop(m)


# -----------------------
# Ultrasonic Helpers
# -----------------------
def single_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    start = time.time()
    timeout = start + 0.03
    while GPIO.input(echo) == 0:
        start = time.time()
        if time.time() > timeout:
            return None

    stop = time.time()
    timeout = stop + 0.03
    while GPIO.input(echo) == 1:
        stop = time.time()
        if time.time() > timeout:
            return None

    return (stop - start) * 34300 / 2


def get_distance(t, e):
    readings = []
    for _ in range(SAMPLE_COUNT):
        d = single_distance(t, e)
        if d:
            readings.append(d)
        time.sleep(0.01)

    if not readings:
        return None
    return statistics.median(readings)


# -----------------------
# Laptop STOP/RESUME listener
# -----------------------
REMOTE_COMMAND = "RESUME"


def command_listener():
    global REMOTE_COMMAND

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 9000))
    s.listen(1)
    print("[INFO] Command server active on port 9000.")

    while True:
        conn, addr = s.accept()
        data = conn.recv(64).decode().strip().upper()
        if data in ("STOP", "RESUME"):
            REMOTE_COMMAND = data
            print(f"[REMOTE] Command received: {data}")
        conn.close()


threading.Thread(target=command_listener, daemon=True).start()


# -----------------------
# MAIN LOOP
# -----------------------
def main():
    print("Starting autonomous navigation...")

    # Start camera first
    start_camera()
    time.sleep(1.0)

    enable_driver()
    all_forward(FORWARD_SPEED)

    prev_remote = REMOTE_COMMAND  # track previous command for transition detection

    try:
        while True:

            # detect STOP -> RESUME transition and resume motion
            if prev_remote == "STOP" and REMOTE_COMMAND == "RESUME":
                print("[REMOTE] RESUME received — restarting motors")
                enable_driver()
                all_forward(FORWARD_SPEED)

            prev_remote = REMOTE_COMMAND

            if REMOTE_COMMAND == "STOP":
                # when STOP is active, keep motors stopped
                all_stop()
                time.sleep(0.1)
                continue

            f = get_distance(SENSORS["Front"]["trig"], SENSORS["Front"]["echo"])
            l = get_distance(SENSORS["Left"]["trig"],  SENSORS["Left"]["echo"])
            r = get_distance(SENSORS["Right"]["trig"], SENSORS["Right"]["echo"])

            f = f if f else 999
            l = l if l else 999
            r = r if r else 999

            if f <= EMERGENCY_DIST:
                all_stop()
                print("EMERGENCY STOP!")
                continue

            if f <= SAFE_DIST:
                all_stop()
                time.sleep(0.05)

                all_backward(BACK_SPEED)
                time.sleep(BACK_TIME)
                all_stop()

                direction = "right" if l < r else "left"
                print("Turning", direction)

                if direction == "right":
                    motor_forward("FL", TURN_SPEED)
                    motor_forward("RL", TURN_SPEED)
                    motor_backward("FR", TURN_SPEED)
                    motor_backward("RR", TURN_SPEED)
                else:
                    motor_backward("FL", TURN_SPEED)
                    motor_backward("RL", TURN_SPEED)
                    motor_forward("FR", TURN_SPEED)
                    motor_forward("RR", TURN_SPEED)

                time.sleep(TURN_TIME)
                all_stop()
                all_forward(FORWARD_SPEED)

            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        all_stop()
        disable_driver()
        GPIO.cleanup()
        stop_camera()
        print("Clean exit.")


if __name__ == "__main__":
    main()