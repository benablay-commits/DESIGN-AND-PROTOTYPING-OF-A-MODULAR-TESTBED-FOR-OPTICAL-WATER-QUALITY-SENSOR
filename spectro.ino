#include <Wire.h>
#include <Adafruit_MCP9808.h>
#include <SparkFun_AS7265X.h>

#include "TLC59108Board.h"

static const int I2C_SDA_PIN = 21;
static const int I2C_SCL_PIN = 22;
static const uint32_t SERIAL_BAUD = 921600;
static const uint32_t DEFAULT_STREAM_INTERVAL_MS = 25;
static const uint8_t MCP9808_I2C_ADDRESS = 0x18;
static const int PELTIER_PWM_PIN = 32;  // H-bridge ENA
static const int PELTIER_IN1_PIN = 33;  // H-bridge EN1
static const int PELTIER_IN2_PIN = 25;  // H-bridge EN2
static const int VIBRATOR_PWM_PIN = 14; // H-bridge ENB
static const int VIBRATOR_IN1_PIN = 26; // H-bridge EN3
static const int VIBRATOR_IN2_PIN = 27; // H-bridge EN4
static const uint32_t PELTIER_PWM_FREQ = 20000;
static const uint32_t VIBRATOR_PWM_FREQ = 500;
static const uint8_t ACTUATOR_PWM_RESOLUTION = 8;
AS7265X spectralSensor;
Adafruit_MCP9808 cuvetteTempSensor = Adafruit_MCP9808();
TLC59108Board ledDriver(Wire, 0x41);

String serialBuffer;
bool streamingEnabled = true;
bool cuvetteTempAvailable = false;
uint32_t streamIntervalMs = DEFAULT_STREAM_INTERVAL_MS;
uint32_t lastStreamAt = 0;
uint8_t sensorGain = 2;                // 0=1x, 1=3.7x, 2=16x, 3=64x
uint8_t sensorIntegrationCycles = 4;   // ~13.9ms
bool peltierCycleEnabled = false;
bool peltierCycleOutputOn = false;
uint8_t peltierCycleMode = 0;          // 1=heat, 2=cool
uint8_t lastPeltierDirection = 0;      // 1=heat, 2=cool; kept after off for safe direction changes
uint8_t peltierCyclePwm = 0;
uint32_t peltierCycleOnMs = 5000;
uint32_t peltierCycleOffMs = 5000;
uint32_t peltierCyclePhaseStartedAt = 0;

const char *CHANNEL_NAMES[18] = {
  "410", "435", "460", "485", "510", "535",
  "560", "585", "610", "645", "680", "705",
  "730", "760", "810", "860", "900", "940"
};

void configureActuators() {
  pinMode(PELTIER_IN1_PIN, OUTPUT);
  pinMode(PELTIER_IN2_PIN, OUTPUT);
  pinMode(VIBRATOR_IN1_PIN, OUTPUT);
  pinMode(VIBRATOR_IN2_PIN, OUTPUT);

  digitalWrite(PELTIER_IN1_PIN, LOW);
  digitalWrite(PELTIER_IN2_PIN, LOW);
  digitalWrite(VIBRATOR_IN1_PIN, LOW);
  digitalWrite(VIBRATOR_IN2_PIN, LOW);

  ledcAttach(PELTIER_PWM_PIN, PELTIER_PWM_FREQ, ACTUATOR_PWM_RESOLUTION);
  ledcAttach(VIBRATOR_PWM_PIN, VIBRATOR_PWM_FREQ, ACTUATOR_PWM_RESOLUTION);
  ledcWrite(PELTIER_PWM_PIN, 0);
  ledcWrite(VIBRATOR_PWM_PIN, 0);
}

void setPeltierOff() {
  digitalWrite(PELTIER_IN1_PIN, LOW);
  digitalWrite(PELTIER_IN2_PIN, LOW);
  ledcWrite(PELTIER_PWM_PIN, 0);
}

void setPeltierHeat(uint8_t pwm) {
  digitalWrite(PELTIER_IN1_PIN, HIGH);
  digitalWrite(PELTIER_IN2_PIN, LOW);
  ledcWrite(PELTIER_PWM_PIN, pwm);
  lastPeltierDirection = 1;
}

void setPeltierCool(uint8_t pwm) {
  digitalWrite(PELTIER_IN1_PIN, LOW);
  digitalWrite(PELTIER_IN2_PIN, HIGH);
  ledcWrite(PELTIER_PWM_PIN, pwm);
  lastPeltierDirection = 2;
}

void applyPeltierCycleOutput() {
  if (!peltierCycleEnabled || !peltierCycleOutputOn || peltierCyclePwm == 0) {
    setPeltierOff();
    return;
  }

  if (peltierCycleMode == 1) {
    setPeltierHeat(peltierCyclePwm);
  } else if (peltierCycleMode == 2) {
    setPeltierCool(peltierCyclePwm);
  } else {
    setPeltierOff();
  }
}

void stopPeltierCycle() {
  peltierCycleEnabled = false;
  peltierCycleOutputOn = false;
  setPeltierOff();
}

void startPeltierCycle(uint8_t mode, uint8_t pwm, uint32_t onMs, uint32_t offMs) {
  const bool directionChanged = lastPeltierDirection != 0 && lastPeltierDirection != mode;
  peltierCycleMode = mode;
  peltierCyclePwm = pwm;
  peltierCycleOnMs = onMs < 100 ? 100 : onMs;
  peltierCycleOffMs = offMs < 100 ? 100 : offMs;
  peltierCycleEnabled = true;
  peltierCycleOutputOn = !directionChanged;
  peltierCyclePhaseStartedAt = millis();
  applyPeltierCycleOutput();

  Serial.print("OK,PELTIER,CYCLE,");
  Serial.print(mode == 1 ? "HEAT" : "COOL");
  Serial.print(",");
  Serial.print(peltierCyclePwm);
  Serial.print(",");
  Serial.print(peltierCycleOnMs);
  Serial.print(",");
  Serial.print(peltierCycleOffMs);
  Serial.print(",");
  Serial.println(directionChanged ? "WAIT_OFF_FIRST" : "START_ON");
}

void updatePeltierCycle(uint32_t now) {
  if (!peltierCycleEnabled) {
    return;
  }

  const uint32_t phaseDuration = peltierCycleOutputOn ? peltierCycleOnMs : peltierCycleOffMs;
  if (now - peltierCyclePhaseStartedAt < phaseDuration) {
    return;
  }

  peltierCycleOutputOn = !peltierCycleOutputOn;
  peltierCyclePhaseStartedAt = now;
  applyPeltierCycleOutput();
  Serial.println(peltierCycleOutputOn ? "STATE,PELTIER,ON" : "STATE,PELTIER,OFF");
}

int clampPwm(String valueText) {
  int pwm = valueText.toInt();
  if (pwm < 0) {
    pwm = 0;
  }
  if (pwm > 255) {
    pwm = 255;
  }
  return pwm;
}

void setVibratorOff() {
  digitalWrite(VIBRATOR_IN1_PIN, LOW);
  digitalWrite(VIBRATOR_IN2_PIN, LOW);
  ledcWrite(VIBRATOR_PWM_PIN, 0);
}

void setVibratorLevel(uint8_t pwm) {
  if (pwm == 0) {
    setVibratorOff();
    return;
  }

  digitalWrite(VIBRATOR_IN1_PIN, HIGH);
  digitalWrite(VIBRATOR_IN2_PIN, LOW);
  ledcWrite(VIBRATOR_PWM_PIN, pwm);
}

void applySensorSettings() {
  spectralSensor.setMeasurementMode(3); // One-shot 6-channel mode
  spectralSensor.setGain(sensorGain);
  spectralSensor.setIntegrationCycles(sensorIntegrationCycles);
}

void sendSensorFrame() {
  spectralSensor.takeMeasurements();

  Serial.print("DATA,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedA(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedB(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedC(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedD(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedE(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedF(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedG(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedH(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedR(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedI(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedS(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedJ(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedT(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedU(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedV(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedW(), 4);
  Serial.print(",");
  Serial.print(spectralSensor.getCalibratedK(), 4);
  Serial.print(",");
  Serial.println(spectralSensor.getCalibratedL(), 4);
}

void sendRawSensorFrame() {
  Serial.print("RAW,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(spectralSensor.getA());
  Serial.print(",");
  Serial.print(spectralSensor.getB());
  Serial.print(",");
  Serial.print(spectralSensor.getC());
  Serial.print(",");
  Serial.print(spectralSensor.getD());
  Serial.print(",");
  Serial.print(spectralSensor.getE());
  Serial.print(",");
  Serial.print(spectralSensor.getF());
  Serial.print(",");
  Serial.print(spectralSensor.getG());
  Serial.print(",");
  Serial.print(spectralSensor.getH());
  Serial.print(",");
  Serial.print(spectralSensor.getR());
  Serial.print(",");
  Serial.print(spectralSensor.getI());
  Serial.print(",");
  Serial.print(spectralSensor.getS());
  Serial.print(",");
  Serial.print(spectralSensor.getJ());
  Serial.print(",");
  Serial.print(spectralSensor.getT());
  Serial.print(",");
  Serial.print(spectralSensor.getU());
  Serial.print(",");
  Serial.print(spectralSensor.getV());
  Serial.print(",");
  Serial.print(spectralSensor.getW());
  Serial.print(",");
  Serial.print(spectralSensor.getK());
  Serial.print(",");
  Serial.println(spectralSensor.getL());
}

void sendTemperatureFrame(bool reportUnavailable = false) {
  if (!cuvetteTempAvailable) {
    if (reportUnavailable) {
      Serial.println("ERR,TEMP_UNAVAILABLE");
    }
    return;
  }

  const float temperatureC = cuvetteTempSensor.readTempC();
  Serial.print("TEMP,");
  Serial.print(millis());
  Serial.print(",");
  Serial.println(temperatureC, 3);
}

void sendHelp() {
  Serial.println("INFO,COMMANDS=PING|STREAM ON|STREAM OFF|ALL OFF|NIR ON|NIR OFF|WHITE ON|WHITE OFF|UV ON|UV OFF|NIR ONLY|WHITE ONLY|UV ONLY|PELTIER OFF|PELTIER HEAT <0-255>|PELTIER COOL <0-255>|PELTIER CYCLE HEAT <0-255> <on_ms> <off_ms>|PELTIER CYCLE COOL <0-255> <on_ms> <off_ms>|PELTIER CYCLE OFF|VIBRATOR OFF|VIBRATOR <0-255>|TEMP|RESET|INTERVAL <ms>|GAIN <0-3>|INTEGRATION <0-255>|HEADER");
}

void sendHeader() {
  Serial.print("HEADER,millis");
  for (uint8_t i = 0; i < 18; ++i) {
    Serial.print(",");
    Serial.print(CHANNEL_NAMES[i]);
  }
  Serial.println();
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "PING") {
    Serial.println("OK,PING");
  } else if (command == "STREAM ON") {
    streamingEnabled = true;
    Serial.println("OK,STREAM ON");
  } else if (command == "STREAM OFF") {
    streamingEnabled = false;
    Serial.println("OK,STREAM OFF");
  } else if (command == "HEADER") {
    sendHeader();
  } else if (command == "TEMP") {
    sendTemperatureFrame(true);
  } else if (command == "ALL OFF") {
    ledDriver.allOff();
    stopPeltierCycle();
    setVibratorOff();
    Serial.println("OK,ALL OFF");
  } else if (command == "NIR ON") {
    ledDriver.setChannel(1, true, 255);
    Serial.println("OK,NIR ON");
  } else if (command == "NIR OFF") {
    ledDriver.setChannel(1, false, 0);
    Serial.println("OK,NIR OFF");
  } else if (command == "WHITE ON") {
    ledDriver.setChannel(2, true, 255);
    Serial.println("OK,WHITE ON");
  } else if (command == "WHITE OFF") {
    ledDriver.setChannel(2, false, 0);
    Serial.println("OK,WHITE OFF");
  } else if (command == "UV ON") {
    ledDriver.setChannel(0, true, 255);
    Serial.println("OK,UV ON");
  } else if (command == "UV OFF") {
    ledDriver.setChannel(0, false, 0);
    Serial.println("OK,UV OFF");
  } else if (command == "NIR ONLY") {
    ledDriver.irOnly(255);
    Serial.println("OK,NIR ONLY");
  } else if (command == "WHITE ONLY") {
    ledDriver.luxeonOnly(255);
    Serial.println("OK,WHITE ONLY");
  } else if (command == "UV ONLY") {
    ledDriver.uvOnly(255);
    Serial.println("OK,UV ONLY");
  } else if (command == "PELTIER OFF") {
    stopPeltierCycle();
    Serial.println("OK,PELTIER,OFF,0");
  } else if (command == "PELTIER CYCLE OFF") {
    stopPeltierCycle();
    Serial.println("OK,PELTIER,OFF");
  } else if (command.startsWith("PELTIER CYCLE ")) {
    String cycleArgs = command.substring(14);
    int firstSpace = cycleArgs.indexOf(' ');
    int secondSpace = cycleArgs.indexOf(' ', firstSpace + 1);
    int thirdSpace = cycleArgs.indexOf(' ', secondSpace + 1);

    if (firstSpace < 0 || secondSpace < 0 || thirdSpace < 0) {
      Serial.println("ERR,PELTIER_CYCLE");
      return;
    }

    String cycleModeText = cycleArgs.substring(0, firstSpace);
    int requestedPwm = clampPwm(cycleArgs.substring(firstSpace + 1, secondSpace));
    uint32_t requestedOnMs = cycleArgs.substring(secondSpace + 1, thirdSpace).toInt();
    uint32_t requestedOffMs = cycleArgs.substring(thirdSpace + 1).toInt();

    if (requestedPwm == 0) {
      stopPeltierCycle();
      Serial.println("OK,PELTIER,OFF");
      return;
    }

    uint8_t cycleMode = 0;
    if (cycleModeText == "HEAT") {
      cycleMode = 1;
    } else if (cycleModeText == "COOL") {
      cycleMode = 2;
    } else {
      Serial.println("ERR,PELTIER_CYCLE_MODE");
      return;
    }

    startPeltierCycle(cycleMode, static_cast<uint8_t>(requestedPwm), requestedOnMs, requestedOffMs);
  } else if (command.startsWith("PELTIER HEAT ")) {
    peltierCycleEnabled = false;
    int requestedPwm = clampPwm(command.substring(13));

    if (requestedPwm == 0) {
      setPeltierOff();
      Serial.println("OK,PELTIER,OFF,0");
    } else {
      setPeltierHeat(static_cast<uint8_t>(requestedPwm));
      Serial.print("OK,PELTIER,HEAT,");
      Serial.println(requestedPwm);
    }
  } else if (command.startsWith("PELTIER COOL ")) {
    peltierCycleEnabled = false;
    int requestedPwm = clampPwm(command.substring(13));

    if (requestedPwm == 0) {
      setPeltierOff();
      Serial.println("OK,PELTIER,OFF,0");
    } else {
      setPeltierCool(static_cast<uint8_t>(requestedPwm));
      Serial.print("OK,PELTIER,COOL,");
      Serial.println(requestedPwm);
    }
  } else if (command == "VIBRATOR OFF") {
    setVibratorOff();
    Serial.println("OK,VIBRATOR,OFF,0");
  } else if (command.startsWith("VIBRATOR ")) {
    int requestedPwm = command.substring(10).toInt();
    if (requestedPwm < 0) {
      requestedPwm = 0;
    }
    if (requestedPwm > 255) {
      requestedPwm = 255;
    }

    setVibratorLevel(static_cast<uint8_t>(requestedPwm));
    Serial.print("OK,VIBRATOR,ON,");
    Serial.println(requestedPwm);
  } else if (command.startsWith("INTERVAL ")) {
    uint32_t newInterval = command.substring(9).toInt();
    if (newInterval < 5) {
      newInterval = 5;
    }
    streamIntervalMs = newInterval;
    Serial.print("OK,INTERVAL,");
    Serial.println(streamIntervalMs);
  } else if (command.startsWith("GAIN ")) {
    int newGain = command.substring(5).toInt();
    if (newGain < 0 || newGain > 3) {
      Serial.println("ERR,GAIN");
      return;
    }

    sensorGain = static_cast<uint8_t>(newGain);
    applySensorSettings();
    Serial.print("OK,GAIN,");
    Serial.println(sensorGain);
  } else if (command.startsWith("INTEGRATION ")) {
    int newCycles = command.substring(12).toInt();
    if (newCycles < 0) {
      newCycles = 0;
    }
    if (newCycles > 255) {
      newCycles = 255;
    }

    sensorIntegrationCycles = static_cast<uint8_t>(newCycles);
    applySensorSettings();
    Serial.print("OK,INTEGRATION,");
    Serial.println(sensorIntegrationCycles);
  } else if (command == "RESET") {
    Serial.println("OK,RESET");
    delay(50);
    ESP.restart();
  } else if (command == "HELP") {
    sendHelp();
  } else {
    Serial.print("ERR,UNKNOWN,");
    Serial.println(command);
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);
  configureActuators();

  if (!ledDriver.begin()) {
    Serial.println("ERR,LED_DRIVER_INIT");
    while (true) {
      delay(1000);
    }
  }

  if (!spectralSensor.begin()) {
    Serial.println("ERR,AS7265X_INIT");
    while (true) {
      delay(1000);
    }
  }

  if (cuvetteTempSensor.begin(MCP9808_I2C_ADDRESS)) {
    cuvetteTempSensor.setResolution(3);
    cuvetteTempAvailable = true;
    Serial.println("READY,MCP9808");
  } else {
    cuvetteTempAvailable = false;
    Serial.println("ERR,MCP9808_INIT");
  }

  spectralSensor.disableIndicator();
  applySensorSettings();
  ledDriver.allOff();
  setPeltierOff();
  setVibratorOff();

  sendHeader();
  Serial.println("READY,AS7265X");
}

void loop() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        processCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }

  const uint32_t now = millis();
  updatePeltierCycle(now);

  if (streamingEnabled && (now - lastStreamAt >= streamIntervalMs)) {
    lastStreamAt = now;
    sendSensorFrame();
    sendRawSensorFrame();
    sendTemperatureFrame(false);
  }
}
