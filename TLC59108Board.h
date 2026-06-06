#ifndef TLC59108BOARD_H
#define TLC59108BOARD_H

#include <Arduino.h>
#include <Wire.h>

class TLC59108Board {
public:
  TLC59108Board(TwoWire &wire = Wire, uint8_t address = 0x41);

  bool begin();
  bool isConnected();

  bool setUV(uint8_t value);
  bool setIR(uint8_t value);
  bool setLuxeon(uint8_t value);

  bool uvOnly(uint8_t value = 255);
  bool irOnly(uint8_t value = 255);
  bool luxeonOnly(uint8_t value = 255);
  bool allOff();

  // Individuele kanaalcontrole zonder andere kanalen te beinvloeden
  bool setChannel(uint8_t ch, bool on, uint8_t pwm = 255);

  void testSequence(uint16_t delayMs = 1500);

private:
  TwoWire *_wire;
  uint8_t _addr;

  // Bijhouden van huidige staat per kanaal
  uint8_t _chMode[3]; // 0x00=OFF, 0x02=PWM
  uint8_t _chPWM[3];

  static const uint8_t REG_MODE1 = 0x00;
  static const uint8_t REG_MODE2 = 0x01;
  static const uint8_t REG_PWM0 = 0x02;
  static const uint8_t REG_GRPPWM = 0x0A;
  static const uint8_t REG_GRPFREQ = 0x0B;
  static const uint8_t REG_LEDOUT0 = 0x0C;
  static const uint8_t REG_IREF = 0x11;

  static const uint8_t CH_UV = 0;      // LED0 = VLMU
  static const uint8_t CH_IR = 1;      // LED1 = SIR19
  static const uint8_t CH_LUXEON = 2;  // LED2 = Luxeon

  bool writeRegister(uint8_t reg, uint8_t value);
  bool readRegister(uint8_t reg, uint8_t &value);
  bool setPWM(uint8_t channel, uint8_t value);

  bool applyMainLedModes(uint8_t uvMode, uint8_t irMode, uint8_t luxMode);
};

#endif
