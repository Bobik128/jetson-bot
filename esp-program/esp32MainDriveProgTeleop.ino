// ============================================================
// ESP32 Robot Drive Firmware (ESP32 lib version=3.3.2)
// Velocity-control version (recommended for teleop + IL)
//
// Jetson sends on Serial2 (UART):
//   CMD V=<linear_m_s> W=<angular_rad_s>
//
// Example:
//   CMD V=0.200 W=-0.500
//
// Meaning:
//   V = forward chassis velocity in m/s (+ forward)
//   W = yaw rate in rad/s (+ CCW / turn left)
//
// ESP32:
//   - Converts (V,W) -> left/right wheel angular velocity targets [rad/s]
//   - Runs both BLDC motors in MotionControlType::velocity
//   - Continuously calls loopFOC() + move(target_omega)
//   - Keeps last commanded target until a new CMD arrives
//   - Sends telemetry lines back over Serial2:
//       STATE vx=<V> vw=<W> batt=<volts>
//
// Also:
//   - PID gains and output_ramp can be tuned live over USB Serial Commander
//   - The tuned values are saved to ESP32 flash (NVS)
//   - On next boot, they are restored automatically
//
// ============================================================

#include <Arduino.h>
#include <SimpleFOC.h>
#include <Wire.h>
#include <Preferences.h>   // <-- persistent storage on ESP32 flash

// ------------------ Robot geometry ------------------
// Tune these to match YOUR robot:
static const float TRACK_WIDTH_M   = 0.160f;     // distance between wheel contact centers [m]
static const float WHEEL_DIAM_M    = 0.0903f;    // wheel diameter [m]
static const float WHEEL_RADIUS_M  = WHEEL_DIAM_M / 2.0f;

// Safety/output limits
static const float SUPPLY_VOLTAGE  = 12.0f;
static const float VOLTAGE_LIMIT   = 8.0f;       // keep conservative initially
static const float CURRENT_LIMIT   = 2.0f;       // amps, tune if too weak
static const float VEL_LIMIT_RAD_S = 40.0f;      // rad/s max wheel speed allowed by controller

// ------------------ Serial config -------------------
static const uint32_t USB_BAUD      = 115200;    // Serial (USB) for debug/Commander
static const uint32_t JETSON_BAUD   = 115200;    // Serial2 for Jetson
static const int PIN_RX2            = 16;        // ESP32 RX2 pin wired to Jetson TX
static const int PIN_TX2            = 17;        // ESP32 TX2 pin wired to Jetson RX

// ------------------ Control state -------------------
// These are what Jetson last commanded, persistent across loops.
volatile float cmd_v_mps = 0.0f;    // forward velocity [m/s]
volatile float cmd_w_rps = 0.0f;    // yaw rate [rad/s]

// We'll convert these every loop into per-wheel angular velocity targets
volatile float target_left_rad_s  = 0.0f;
volatile float target_right_rad_s = 0.0f;

// -------- Persistent tunables (mirrored to motors) --------
// These act as our "authoritative" runtime config.
// They get loaded from flash on boot (if available)
// and written back to flash when changed by Commander.
float pidP     = 0.3f;
float pidI     = 10.0f;
float pidD     = 0.002f;
float pidRamp  = 60.0f;

// Preferences instance for NVS
Preferences prefs;
static const char* NVS_NAMESPACE = "drivePID";

// ------------------ Sensors -------------------------
// Two AS5600 magnetic sensors, one per motor.
// You had one on Wire, one on Wire1.
// We'll keep that dual bus layout.

#define USE_DUAL_I2C 1

MagneticSensorI2C sensorLeft  = MagneticSensorI2C(AS5600_I2C);
MagneticSensorI2C sensorRight = MagneticSensorI2C(AS5600_I2C);

// ------------------ Motors & drivers ----------------
// Your original pinout:
//
//  driverLeft  = BLDCDriver3PWM(13, 12, 14, 5);
//  driverRight = BLDCDriver3PWM(33, 25, 26, 18);
//  motorLeft   = BLDCMotor(11);
//  motorRight  = BLDCMotor(11);
//
// 11 = number of pole pairs (adjust if your motors differ)

BLDCMotor motorLeft  = BLDCMotor(11);   // LEFT motor
BLDCMotor motorRight = BLDCMotor(11);   // RIGHT motor

InlineCurrentSense currentSenseLeft  = InlineCurrentSense(100.0, 34, 35, 32);
InlineCurrentSense currentSenseRight  = InlineCurrentSense(100.0, 27, 2, 4);

BLDCDriver3PWM driverLeft  = BLDCDriver3PWM(13, 12, 14, 5);
BLDCDriver3PWM driverRight = BLDCDriver3PWM(33, 25, 26, 18);

// ------------------ Commander for tuning -------------
Commander commander = Commander(Serial);

// Forward declarations of commander callbacks
void cmdSetVelP(char* cmd);
void cmdSetVelI(char* cmd);
void cmdSetVelD(char* cmd);
void cmdSetVelRamp(char* cmd);
void cmdPrintPID(char* cmd);

// ------------------ Persistent config helpers --------
//
// Push the current pidP/pidI/pidD/pidRamp values into both motors.
// Call this after loading from flash OR after tuning.
void applyPIDToMotors() {
  motorLeft.PID_velocity.P  = pidP;
  motorLeft.PID_velocity.I  = pidI;
  motorLeft.PID_velocity.D  = pidD;
  motorLeft.PID_velocity.output_ramp = pidRamp;

  motorRight.PID_velocity.P  = pidP;
  motorRight.PID_velocity.I  = pidI;
  motorRight.PID_velocity.D  = pidD;
  motorRight.PID_velocity.output_ramp = pidRamp;
}

// Load PID/ramp from flash, if available.
void loadPIDFromFlash() {
  if (!prefs.begin(NVS_NAMESPACE, /*readOnly=*/true)) {
    Serial.println("NVS open (RO) failed, using defaults.");
    return;
  }

  pidP    = prefs.getFloat("P",    pidP);
  pidI    = prefs.getFloat("I",    pidI);
  pidD    = prefs.getFloat("D",    pidD);
  pidRamp = prefs.getFloat("ramp", pidRamp);

  prefs.end();

  Serial.println("Loaded PID from NVS:");
  Serial.printf("  P=%.4f I=%.4f D=%.6f ramp=%.3f\n", pidP, pidI, pidD, pidRamp);
}

// Save PID/ramp to flash (called after tuning via Commander)
void savePIDToFlash() {
  if (!prefs.begin(NVS_NAMESPACE, /*readOnly=*/false)) {
    Serial.println("NVS open (RW) failed, NOT saved!");
    return;
  }

  prefs.putFloat("P",    pidP);
  prefs.putFloat("I",    pidI);
  prefs.putFloat("D",    pidD);
  prefs.putFloat("ramp", pidRamp);

  prefs.end();

  Serial.println("Saved PID to NVS:");
  Serial.printf("  P=%.4f I=%.4f D=%.6f ramp=%.3f\n", pidP, pidI, pidD, pidRamp);
}

// ------------------ Commander callback impl ----------
//
// All of these now:
//  1) parse new value
//  2) update our persistent vars
//  3) apply to motors
//  4) save to flash

void cmdSetVelP(char* cmd) {
  float v = atof(cmd);
  pidP = v;
  applyPIDToMotors();
  savePIDToFlash();
  Serial.printf("New P=%.4f\n", pidP);
}

void cmdSetVelI(char* cmd) {
  float v = atof(cmd);
  pidI = v;
  applyPIDToMotors();
  savePIDToFlash();
  Serial.printf("New I=%.4f\n", pidI);
}

void cmdSetVelD(char* cmd) {
  float v = atof(cmd);
  pidD = v;
  applyPIDToMotors();
  savePIDToFlash();
  Serial.printf("New D=%.6f\n", pidD);
}

void cmdSetVelRamp(char* cmd) {
  float v = atof(cmd);
  pidRamp = v;
  applyPIDToMotors();
  savePIDToFlash();
  Serial.printf("New ramp=%.3f\n", pidRamp);
}

void cmdPrintPID(char* cmd) {
  (void)cmd;
  Serial.printf("PID persisted: P=%.3f I=%.3f D=%.6f ramp=%.3f\n",
    pidP, pidI, pidD, pidRamp);

  Serial.printf("Motor L: P=%.3f I=%.3f D=%.6f ramp=%.3f | "
                "Motor R: P=%.3f I=%.3f D=%.6f ramp=%.3f\n",
    motorLeft.PID_velocity.P,  motorLeft.PID_velocity.I,
    motorLeft.PID_velocity.D,  motorLeft.PID_velocity.output_ramp,
    motorRight.PID_velocity.P, motorRight.PID_velocity.I,
    motorRight.PID_velocity.D, motorRight.PID_velocity.output_ramp);
}

// ------------------ Serial2 parsing ------------------
//
// Expected format exactly:
//   CMD V=<float> W=<float>
//
// Example:
//   CMD V=0.250 W=-0.700
//
// This parser runs in loop() and updates cmd_v_mps / cmd_w_rps.

String rx2Line;
uint32_t bytes_rx2  = 0;
uint32_t lines_rx2  = 0;
uint32_t last_dbgms = 0;

void applyVelocityCommand(const String& line) {
  float v_in = 0.0f;
  float w_in = 0.0f;

  if (!line.startsWith("CMD")) return;

  int parsed = sscanf(line.c_str(), "CMD V=%f W=%f", &v_in, &w_in);
  if (parsed == 2) {
    cmd_v_mps = v_in;
    cmd_w_rps = w_in;
  }
}

void pollJetsonSerial() {
  while (Serial2.available()) {
    char c = (char)Serial2.read();
    bytes_rx2++;

    if (c == '\n' || c == '\r') {
      if (rx2Line.length() > 0) {
        lines_rx2++;
        // debug echo to USB
        // Serial.print("[RX2 LINE] ");
        // Serial.println(rx2Line);
        applyVelocityCommand(rx2Line);
        rx2Line = "";
      }
    } else {
      if (rx2Line.length() < 100) {
        rx2Line += c;
      } else {
        rx2Line = ""; // overflow guard
      }
    }
  }

  // uint32_t now = millis();
  // if (now - last_dbgms > 60) {
  //   PhaseCurrent_s currents = currentSenseRight.getPhaseCurrents();
  //   Serial.print(currents.a);
  //   Serial.print(", ");
  //   Serial.print(currents.b);
  //   Serial.print(", ");
  //   Serial.print(currents.c);
  //   Serial.print(", ");

  //   PhaseCurrent_s currents2 = currentSenseLeft.getPhaseCurrents();
  //   Serial.print(currents2.a);
  //   Serial.print(", ");
  //   Serial.print(currents2.b);
  //   Serial.print(", ");
  //   Serial.println(currents2.c);
  //   last_dbgms = now;
  // }
  

  // print stats every 500ms
  // uint32_t now = millis();
  // if (now - last_dbgms > 500) {
  //   last_dbgms = now;
  //   Serial.print("[RX2 STATS] bytes=");
  //   Serial.print(bytes_rx2);
  //   Serial.print(" lines=");
  //   Serial.print(lines_rx2);
  //   Serial.print(" last_v=");
  //   Serial.print(cmd_v_mps, 3);
  //   Serial.print(" last_w=");
  //   Serial.print(cmd_w_rps, 3);
  //   Serial.println();
  // }
}

// ------------------ Kinematics ------------------------
//
// Convert commanded chassis twist (v,w) into per-wheel angular velocities.
// Differential drive:
//   v_left  = v - w*(track_width/2)
//   v_right = v + w*(track_width/2)
// Then wheel_omega = linear / wheel_radius
//
// Signs might need flipping depending on wiring.
// There's guidance below.

void computeWheelTargets() {
  float v = cmd_v_mps;
  float w = cmd_w_rps;

  float v_left  = v - w * (TRACK_WIDTH_M * 0.5f);
  float v_right = v + w * (TRACK_WIDTH_M * 0.5f);

  float left_omega  = v_left  / WHEEL_RADIUS_M; // rad/s
  float right_omega = v_right / WHEEL_RADIUS_M; // rad/s

  // If forward stick drives backward, flip both:
  // left_omega  = -left_omega;
  // right_omega = -right_omega;
  //
  // If it spins instead of translating, swap:
  // float tmp = left_omega; left_omega = right_omega; right_omega = tmp;
  //
  // Or flip just one side if turns are reversed:
  // left_omega = -left_omega;

  // clamp for safety
  if (left_omega  >  VEL_LIMIT_RAD_S) left_omega  =  VEL_LIMIT_RAD_S;
  if (left_omega  < -VEL_LIMIT_RAD_S) left_omega  = -VEL_LIMIT_RAD_S;
  if (right_omega >  VEL_LIMIT_RAD_S) right_omega =  VEL_LIMIT_RAD_S;
  if (right_omega < -VEL_LIMIT_RAD_S) right_omega = -VEL_LIMIT_RAD_S;

  target_left_rad_s  = left_omega;
  target_right_rad_s = right_omega;
}

// ------------------ Telemetry back to Jetson ---------

void sendTelemetry() {
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if (now - last_ms < 30) return;
  last_ms = now;

  float left_omega = sensorLeft.getVelocity();
  float right_omega = -sensorRight.getVelocity();

  float v_left  = left_omega  * WHEEL_RADIUS_M;
  float v_right = right_omega * WHEEL_RADIUS_M;

  float v = 0.5f * (v_left + v_right);                 // m/s
  float w = (v_right - v_left) / TRACK_WIDTH_M;        // rad/s

  Serial2.print(" v=");
  Serial2.print(v, 3);
  Serial2.print(" w=");
  Serial2.print(w, 3);
  Serial2.print("\n");
}

// ------------------ FOC control task -----------------
//
// Runs high-rate motor control loop in its own FreeRTOS task.
// Both motors in MotionControlType::velocity.
// We continuously call loopFOC() and move(rad/s_target).
//
// We KEEP MOVING at the last commanded velocity even if UART is silent.

void focTask(void* pvParameters) {
  (void)pvParameters;
  for (;;) {
    // update wheel targets from last cmd_v/cmd_w
    computeWheelTargets();

    // run FOC loops
    motorLeft.loopFOC();
    motorRight.loopFOC();

    // send velocity targets (rad/s)
    //
    // NOTE: sign on right motor is sometimes inverted depending on how
    // phases are wired physically. Keep / remove the minus on right as needed.
    motorLeft.move(target_left_rad_s);
    motorRight.move(-target_right_rad_s);

    // allow commander tuning on USB without starving control
    commander.run();

    // Yield so FreeRTOS can schedule others
    taskYIELD();
  }
}

// ------------------ setup() ---------------------------
void setup() {
  // USB serial for debug + Commander
  Serial.begin(USB_BAUD);
  delay(200);

  // Jetson UART link
  Serial2.begin(JETSON_BAUD, SERIAL_8N1, PIN_RX2, PIN_TX2);
  delay(200);

  Serial.println("ESP32 velocity-drive firmware booting...");

#if USE_DUAL_I2C
  // Dual I2C bus setup
  Wire.setClock(400000);
  Wire.begin(); // default SDA/SCL

  Wire1.setClock(400000);
  // SDA=19, SCL=23  (adjust to your wiring)
  Wire1.begin(19, 23, 400000);
#else
  Wire.setClock(400000);
  Wire.begin();
#endif

  Serial.println("Init AS5600 sensors...");
#if USE_DUAL_I2C
  sensorLeft.init();        // default Wire
  sensorRight.init(&Wire1); // second bus
#else
  sensorLeft.init();
  sensorRight.init();
#endif
  Serial.println("Sensors OK.");

  // Attach sensors to motors
  motorLeft.linkSensor(&sensorLeft);
  motorRight.linkSensor(&sensorRight);

  // Init drivers
  driverLeft.voltage_power_supply  = SUPPLY_VOLTAGE;
  driverRight.voltage_power_supply = SUPPLY_VOLTAGE;
  driverLeft.init();
  driverRight.init();

  // Attach drivers
  motorLeft.linkDriver(&driverLeft);
  motorRight.linkDriver(&driverRight);

  // FOC modulation
  motorLeft.foc_modulation  = FOCModulationType::SpaceVectorPWM;
  motorRight.foc_modulation = FOCModulationType::SpaceVectorPWM;

  // VELOCITY MODE
  motorLeft.controller  = MotionControlType::velocity;
  motorRight.controller = MotionControlType::velocity;

  // LINK CURRENT SENSE
  Serial.println("Link current sense");
  currentSenseLeft.linkDriver(&driverLeft);
  currentSenseRight.linkDriver(&driverRight);
  Serial.println("Current sense linked");

  // init current sense
  if (currentSenseLeft.init())  Serial.println("Current sense left init success!");
  else{
    Serial.println("Current sense left init failed!");
    return;
  }
  if (currentSenseRight.init())  Serial.println("Current sense right init success!");
  else{
    Serial.println("Current sense right init failed!");
    return;
  }

  // Load PID/ramp from flash (if present) and apply to motors
  loadPIDFromFlash();
  applyPIDToMotors();

  // Safety limits
  motorLeft.voltage_limit  = VOLTAGE_LIMIT;
  motorRight.voltage_limit = VOLTAGE_LIMIT;
  motorLeft.current_limit  = CURRENT_LIMIT;
  motorRight.current_limit = CURRENT_LIMIT;

  motorLeft.velocity_limit  = VEL_LIMIT_RAD_S;
  motorRight.velocity_limit = VEL_LIMIT_RAD_S;

  // Init motors
  Serial.println("Init motors...");
  motorLeft.init();
  motorRight.init();

  Serial.println("initFOC...");
  motorLeft.initFOC();
  motorRight.initFOC();
  Serial.println("FOC OK.");

  // Commander bindings
  // Usage over USB Serial:
  //   P0.4     -> set P gain = 0.4   (and save)
  //   I8       -> set I gain = 8
  //   D0.0015  -> set D gain = 0.0015
  //   V50      -> set output_ramp = 50
  //   Q        -> print current values
  commander.add('P', cmdSetVelP,    "vel kP");
  commander.add('I', cmdSetVelI,    "vel kI");
  commander.add('D', cmdSetVelD,    "vel kD");
  commander.add('V', cmdSetVelRamp, "vel ramp");
  commander.add('Q', cmdPrintPID,   "print PID");

  Serial.println("READY. Waiting for Jetson 'CMD V=.. W=..'.");

  // Start control task
  xTaskCreate(
    focTask,
    "focTask",
    8192,
    NULL,
    1,
    NULL
  );
}

// ------------------ loop() ----------------------------
void loop() {
  // 1. Read new commands from Jetson (non-blocking)
  pollJetsonSerial();

  // 2. Send telemetry back to Jetson
  sendTelemetry();

  // 3. Tiny sleep to not hog idle core
  delay(1);
}