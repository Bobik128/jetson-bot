![Main Image](main_image.jpg)

# JetsonBot Software Setup

This README explains how to build and wire the JetsonBot hardware, configure the Jetson Orin Nano, install LeRobot, install the JetsonBot packages, calibrate the arm, and run teleoperation, dataset recording, training, and policy evaluation.

The setup uses two computers:

- **Jetson host**: the NVIDIA Jetson Orin Nano mounted on the robot.
- **Client PC**: a Linux PC used for teleoperation, recording, training control, and remote access.

The Jetson is expected to run mostly headless, without HDMI connected, so both devices should be connected to the same local network or VPN.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Build the Robot](#build-the-robot)
3. [Wiring](#wiring)
4. [Jetson Initial Setup](#jetson-initial-setup)
5. [Install LeRobot](#install-lerobot)
6. [Install JetsonBot Packages](#install-jetsonbot-packages)
7. [Apply LeRobot Patches](#apply-lerobot-patches)
8. [Connect and Configure the Robot Arm](#connect-and-configure-the-robot-arm)
9. [Calibrate Motor Limits](#calibrate-motor-limits)
10. [Test Teleoperation](#test-teleoperation)
11. [Record Datasets](#record-datasets)
12. [Train a Policy](#train-a-policy)
13. [Evaluate a Policy](#evaluate-a-policy)
14. [Troubleshooting Notes](#troubleshooting-notes)

---

## Hardware Requirements

### Printed and Mechanical Parts

- Printed parts from the Onshape document:

  ```text
  https://cad.onshape.com/documents/1c027469138e22df9abe4045/w/c7a37744e0032bf1c17d6366/e/4006b5648b1eac5f03fa06ec?renderMode=0&uiState=6a1c07ff284022db124a85fb
  ```

- Acrylic case for the Jetson
- M2 and M3 screws and nuts
- Two 20 mm balls for the omniwheels
- Magnet adapters for the motor hollow shafts, matched to your magnet size

### Electronics

- [Waveshare UPS Module 3S](https://www.waveshare.com/wiki/UPS_Module_3S)
- ESP32, 30-pin version
- MPU6050 IMU
- 2× AS5600 magnetic encoders with magnets
- 2× [SimpleFOC Mini](https://docs.simplefoc.com/simplefocmini) drivers
- 2× [GM4108 motors](https://shop.iflight.com/ipower-motor-gm4108h-120t-brushless-gimbal-motor-pro217)
- 3× Li-Ion 18650 batteries
- 4× [Feetech ST3215](https://www.seeedstudio.com/Feetech-ST-3215-C044-Heavy-Duty-Servo-7-4V-1-191-Gear-Reduction-p-6460.html) servos
  - If you use the 7.4 V version, use an appropriate step-down converter.
- [Bus servo driver board / serial converter](https://www.seeedstudio.com/Bus-Servo-Driver-Board-for-XIAO-p-6413.html) for STS servos

### Computing and Control

- NVIDIA Jetson Orin Nano
- Ubuntu with JetPack 6.2 installed on the Jetson
- Linux PC
- SO101 Leader arm connected to the client PC
- JetsonBot follower arm connected to the Jetson
- Gamepad connected to the client PC

### Cameras

CSI cameras connected to the Jetson, for example:

- IMX477
- IMX219

---

## Build the Robot

Print and buy all parts listed in the [Hardware Requirements](#hardware-requirements) section.

Assembly is fairly simple. Follow the mechanical assembly shown in the Onshape document linked above.

Do not forget to:

- Print the hollow-shaft magnet adapters for your magnet size.
- Place the AS5600 encoders into their slots.
- Make sure the magnets are centered above the encoders.
- Check that the omniwheel balls rotate freely.
- Check that no wires can touch the rotating motors or wheels.

---

## Wiring

The robot contains several connected devices:

- ESP32
- 2× AS5600 magnetic encoders
- 2× SimpleFOC Mini drivers
- MPU6050
- Jetson Orin Nano
- Waveshare UPS module

Make sure all shared devices have a common ground.

---

### AS5600 Encoders

Connect both AS5600 encoders to power first:

| AS5600 Pin | Connect To |
|---|---|
| GND | UPS GND |
| VCC | UPS 3.3 V |

Then connect each encoder to its own ESP32 I2C bus.

#### Left AS5600

| AS5600 Pin | ESP32 Pin |
|---|---|
| SDA | GPIO 21 |
| SCL | GPIO 22 |

#### Right AS5600

| AS5600 Pin | ESP32 Pin |
|---|---|
| SDA | GPIO 19 |
| SCL | GPIO 23 |

---

### SimpleFOC Mini Drivers

Connect both SimpleFOC Mini drivers to power and to the motors.

| SimpleFOC Mini Pin | Connect To |
|---|---|
| VCC | UPS 12.6 V output |
| GND | UPS GND |
| GND | ESP32 GND |
| M1, M2, M3 | Motor phases |

If a motor rotates in the wrong direction, swap any two motor phase wires.

#### Left SimpleFOC Mini

| SimpleFOC Mini Pin | ESP32 Pin |
|---|---|
| IN1 | GPIO 13 |
| IN2 | GPIO 12 |
| IN3 | GPIO 14 |
| EN | GPIO 5 |

#### Right SimpleFOC Mini

| SimpleFOC Mini Pin | ESP32 Pin |
|---|---|
| IN1 | GPIO 33 |
| IN2 | GPIO 25 |
| IN3 | GPIO 26 |
| EN | GPIO 18 |

---

### ESP32 to Jetson UART

Connect the ESP32 to the Jetson UART.

| ESP32 Pin | Jetson Pin |
|---|---|
| GPIO 16, RX | Pin 8, TX |
| GPIO 17, TX | Pin 10, RX |
| GND | Pin 6, GND |

---

### UPS, MPU6050, and Jetson Power/I2C

| Device Pin | Connect To |
|---|---|
| UPS 5 V | MPU6050 VCC |
| UPS GND | MPU6050 GND and Jetson GND |
| UPS SCL | MPU6050 SCL and Jetson pin 5, SCL |
| UPS SDA | MPU6050 SDA and Jetson pin 3, SDA |
| UPS 12.6 V | Jetson 12.6 V input, front barrel connector |

---

## Jetson Initial Setup

Install Ubuntu with JetPack 6.2 on the Jetson Orin Nano.

Follow the official NVIDIA setup guide:

```text
https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/
```

After the Jetson is set up, connect the CSI cameras to the CSI ports.

Then run the Jetson IO configuration tool:

```bash
sudo python /opt/nvidia/jetson-io/jetson-io.py
```

Configure the 24-pin CSI ports according to the cameras you are using.

If needed, follow this video tutorial:

```text
https://www.youtube.com/watch?v=gJPIJ3yxME0
```

---

## Install LeRobot

Install LeRobot by following the official guide:

```text
https://huggingface.co/docs/lerobot/installation
```

After LeRobot is installed, enable the required extras:

```bash
cd lerobot

pip install 'lerobot[feetech]'
pip install 'lerobot[lekiwi]'
```

---

## Install JetsonBot Packages

Run these steps on both:

- the Jetson host
- the Linux client PC

Activate the LeRobot Conda environment:

```bash
conda activate lerobot
```

Clone the JetsonBot repository:

```bash
git clone https://github.com/Bobik128/jetson-bot.git
```

Install the JetsonBot robot package:

```bash
cd jetson-bot/src/lerobot_robot_jetsonbot

pip install -e .
pip install lerobot_robot_jetsonbot
```

Install the DualSense teleoperator package:

```bash
cd ../lerobot_teleoperator_dualsense

pip install -e .
pip install lerobot_teleoperator_dualsense
```

---

## Apply LeRobot Patches

This step is required on the Jetson.

The patches add JetsonBot support to the LeRobot scripts.

Go to your LeRobot repository:

```bash
cd /path/to/lerobot
```

Check that the patches can be applied cleanly:

```bash
git apply --check ../jetson-bot/'lerobot-modifiing patches'/lerobot_record.patch
git apply --check ../jetson-bot/'lerobot-modifiing patches'/lerobot_setup_motors.patch
```

Apply the patches:

```bash
git apply ../jetson-bot/'lerobot-modifiing patches'/lerobot_record.patch
git apply ../jetson-bot/'lerobot-modifiing patches'/lerobot_setup_motors.patch
```

> Note: the folder name `lerobot-modifiing patches` is kept exactly as used in the repository path.

---

## Connect and Configure the Robot Arm

Connect the robot arm serial bus driver board to a USB port on the Jetson.

Find the correct port:

```bash
lerobot-find-port
```

When prompted, disconnect the MotorBus USB cable and press Enter.

Example output:

```text
Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']

Remove the USB cable from your MotorBus and press Enter when done.

[...Disconnect the corresponding leader or follower arm and press Enter...]

The port of this MotorBus is /dev/ttyACM1

Reconnect the USB cable.
```

In this example, the correct port is:

```text
/dev/ttyACM1
```

On Linux, you may need to give temporary access to the USB ports:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Now set up the motor IDs:

```bash
lerobot-setup-motors \
    --robot.type=jetsonbot \
    --robot.port=/dev/ttyACM1
```

Replace `/dev/ttyACM1` with the port found in the previous step.

Follow the instructions printed by the program.

---

## Calibrate Motor Limits

After the motor IDs are configured, calibrate the robot:

```bash
lerobot-calibrate \
    --robot.type=jetsonbot \
    --robot.port=/dev/ttyACM1 \
    --robot.id=jetson-bot
```

Replace `/dev/ttyACM1` with the correct robot port.

This process is technically the same as calibrating an SO101 follower arm.

After this step, the base software setup is complete.

---

## Test Teleoperation

Connect the following to the client PC:

- SO101 Leader arm
- Gamepad

Run the Jetson host script on the Jetson:

```bash
bash jetson-bot/runners/run_jetson_host.sh
```

On the client PC, edit:

```text
jetson-bot/src/scripts/teleoperate.py
```

Set the Jetson IP address:

```python
remote_ip = "10.102.180.119"
```

Replace the example IP address with the actual IP address of your Jetson.

Set the SO101 Leader arm ID:

```python
leader_id = "the_leader"
```

Replace `the_leader` with your actual leader arm ID.

Run teleoperation from the client PC:

```bash
python3 jetson-bot/src/scripts/teleoperate.py
```

A window with camera feeds and joint feedback should appear.

The robot should react to the SO101 Leader arm and the gamepad.

---

## Record Datasets

The `record.py` script works similarly to `teleoperate.py`.

Before recording, make sure you are logged in to your Hugging Face account.

The gamepad buttons are bound to recording actions such as:

- stop recording
- re-record the current episode
- exit early
- control the recording flow

These bindings can be configured inside `record.py`.

Recorded datasets are automatically uploaded to the LeRobot Hub.

---

## Train a Policy

To train a policy, use the `lerobot-train` script.

Follow the official LeRobot training documentation:

```text
https://huggingface.co/docs/lerobot/
```

For this robot, the recommended policy is **ACT**.

Other policies are usually too heavy for the Jetson and may not run properly during evaluation.

---

## Evaluate a Policy

Evaluation runner scripts are located in:

```text
jetson-bot/runners
```

Before running evaluation, update the runner script with:

- Jetson IP address
- Hugging Face model link
- evaluation dataset name
- number of episodes
- FPS
- task description
- robot ID
- other required evaluation settings

Then run the appropriate evaluation runner from the `runners` folder.

---

## Troubleshooting Notes

### USB Permission Errors

If the robot or leader arm cannot access `/dev/ttyACM*`, run:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

This is temporary and may need to be repeated after reconnecting the device or rebooting.

---

### Teleoperation Starts, but the Robot Does Not Move

Check that:

- the Jetson host script is running
- the Jetson and client PC are on the same network or VPN
- `remote_ip` is set to the Jetson IP address
- `leader_id` matches the SO101 Leader arm ID
- the robot arm is connected to the Jetson
- the gamepad is connected to the client PC
- the correct `/dev/ttyACM*` port is used

---

### Camera Feed Does Not Appear

Check that:

- the CSI cameras are connected correctly
- Jetson IO was configured for the correct 24-pin CSI camera ports
- the camera configuration matches the installed cameras
- the Jetson was rebooted after changing Jetson IO settings

---

## Final Checklist

Before running teleoperation or recording, verify that:

- [ ] JetPack 6.2 is installed on the Jetson
- [ ] CSI cameras are configured through Jetson IO
- [ ] LeRobot is installed
- [ ] `lerobot[feetech]` and `lerobot[lekiwi]` extras are installed
- [ ] JetsonBot packages are installed on both machines
- [ ] LeRobot patches are applied on the Jetson
- [ ] Robot motor IDs are configured
- [ ] Robot limits are calibrated
- [ ] Jetson host script is running
- [ ] Client PC has the correct Jetson IP address
- [ ] SO101 Leader arm ID is configured correctly
- [ ] Gamepad is connected