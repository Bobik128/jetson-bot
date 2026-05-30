Software setup

First, you'll need pc with Linux, since you're not going to have hdmi cable connected to jetson all the time and want it wireless
Then you'll need teh Jetson Orin Nano and ubuntu with jetpack 6.2 installed on it, how to do that is written on nvidia official docs here https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/

when you have jetson set up, connect the csi cameras imx477 and imx219 to the csi ports.
then run this script on it and configure there the csi 24pin port to your cameras
sudo python /opt/nvidia/jetson-io/jetson-io.py

follow this tutorial for it if needed https://www.youtube.com/watch?v=gJPIJ3yxME0

when you have that done, you can install lerobot following their guide: https://huggingface.co/docs/lerobot/installation
also enable neede extra features like this:
cd lerobot
pip install 'lerobot[feetech]'
pip install 'lerobot[lekiwi]'


once that is done, you can install this library like this
conda activate lerobot
git clone https://github.com/Bobik128/jetson-bot.git

cd jetson-bot/src/lerobot_robot_jetsonbot
pip install -e .
pip install lerobot_robot_jetsonbot

cd jetson-bot/src/lerobot_teleoperator_dualsense
pip install -e .
pip install lerobot_teleoperator_dualsense

Do this on both jetson and the other pc

Now on the other pc, I assume, that you already have a teleoperator hand like SO101 ready, setup and remember its id (name)

Now on jetson, you also need to do apply patches to the lerobot lib to include jetsonbot
cd ../lerobot
git apply --check ../jetson-bot/'lerobot-modifiing patches'/lerobot_record.patch
git apply --check ../jetson-bot/'lerobot-modifiing patches'/lerobot_setup_motors.patch

Now connect the arm on robot (the saerial bus driver board) to some usb port on the jetson (to not take up much space, it will stay there)

Run the following script to find the port and disconnect the MotorBus when prompted:
lerobot-find-port

On Linux, you might need to give access to the USB ports by running:

sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

Example output:

Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect corresponding leader or follower arm and press Enter...]

The port of this MotorsBus is /dev/ttyACM1
Reconnect the USB cable.

Where the found port is: /dev/ttyACM1 corresponding to your arm.

to setup the motors and set their ids, run 
lerobot-setup-motors \
    --robot.type=jetsonbot \
    --robot.port=/dev/tty.usbmodem585A0076841  # <- paste here the port found at previous step

follow the programs instructions

them to set the motor limits, run
lerobot-calibrate \
    --robot.type=jetsonbot \
    --robot.port=/dev/tty.usbmodem58760431551 \ # <- The port of your robot
    --robot.id=jetson-bot

this process is technically the same as for SO101 follower arm

Congratulations! now you have completely set up the base software

now to test it, connect your SO101 Leader arm to the client pc, and connect also a gamepad to it.
now on jetson run
bash jetson-bot/runners/run_jetson_host.sh
on pc modify the jetson-bot/src/scripts/teleoperate.py
set the remote_ip to the [10.102.180.119] ip of jetson (they have to be on the same local network or connected trough VPN)
set the leader arm [the_leader] id to your id

then on pc run 
python3 jetson-bot/src/scripts/teleoperate.py

a screen with camera feeds and joint feedback should pop up
the robot should react to the leader arm and gamepad

the record.py script works the same way, but you have to be logged in into lerobot account and some buttons on gamepad are bind to stopping recording, rerecording, exiting early and so, it can be configured within the record.py file
with it, you can record datasets and they get automatically uploaded to the lerobot hub

to tran a policy, you have to use the lerobot_train script, how to use it is on their web
you can train for this robot only act policy, any other is too heavy for the jetson and wont run

then in jetson-bot/runners folder, there are runners for the policy evaluation, you also have to change the ip there, and tha model link and dataset name
