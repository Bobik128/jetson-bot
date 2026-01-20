/usr/bin/python3 ../src/run_policy_act_wheels_arm.py \
  --policy-path ../data/act_my_bot_v3_run1/checkpoints/last/pretrained_model \
  --port /dev/ttyTHS1 --baud 115200 \
  --hz 30 --device cpu \
  --debug-dir ../data \
  --front-sensor-id 0 --side-sensor-id 1 \
  --capture-width 640 --capture-height 480 --capture-fps 30 \
  --arm-enable \
  --arm-port /dev/ttyACM0 --arm-baudrate 1000000 \
  --follower_calib ../data/calibrations/hand_calibration.json