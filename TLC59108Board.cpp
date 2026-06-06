#include "TLC59108Board.h"

TLC59108Board::TLC59108Board(TwoWire &wire, uint8_t address)
    : _wire(&wire), _addr(address) {
  for (uint8_t i = 0; i < 3; i++) {
    _chMode[i] = 0x00;
    _chPWM[i] = 0;
  }
}

bool TLC59108Board::writeRegister(uint8_t reg, uint8_t value) {
  _wire->beginTransmission(_addr);
  _wire->write(reg);
  _wire->write(value);
  return (_wire->endTransmission() == 0);
}

bool TLC59108Board::readRegister(uint8_t reg, uint8_t &value) {
  _wire->beginTransmission(_addr);
  _wire->write(reg);
  if (_wire->endTransmission(false) != 0) return false;

  if (_wire->requestFrom((int)_addr, 1) != 1) return false;
  value = _wire->read();
  return true;
}

bool TLC59108Board::isConnected() {
  _wire->beginTransmission(_addr);
  return (_wire->endTransmission() == 0);
}

bool TLC59108Board::begin() {
  if (!isConnected()) return false;

  if (!writeRegister(REG_MODE1, 0x00)) return false;
  if (!writeRegister(REG_MODE2, 0x00)) return false;

  // groepsdimming uit
  if (!writeRegister(REG_GRPPWM, 0xFF)) return false;
  if (!writeRegister(REG_GRPFREQ, 0x00)) return false;

  // maximale software current reference
  if (!writeRegister(REG_IREF, 0xFF)) return false;

  // alle PWM registers op 0
  for (uint8_t ch = 0; ch < 8; ch++) {
    if (!writeRegister(REG_PWM0 + ch, 0x00)) return false;
  }

  return allOff();
}

bool TLC59108Board::setPWM(uint8_t channel, uint8_t value) {
  if (channel > 7) return false;
  return writeRegister(REG_PWM0 + channel, value);
}

bool TLC59108Board::applyMainLedModes(uint8_t uvMode, uint8_t irMode, uint8_t luxMode) {
  if (uvMode > 0x03 || irMode > 0x03 || luxMode > 0x03) return false;

  uint8_t ledout = 0;
  ledout |= (uvMode << 0);   // LED0 bits 1:0
  ledout |= (irMode << 2);   // LED1 bits 3:2
  ledout |= (luxMode << 4);  // LED2 bits 5:4

  return writeRegister(REG_LEDOUT0, ledout);
}

bool TLC59108Board::setUV(uint8_t value) {
  return setPWM(CH_UV, value);
}

bool TLC59108Board::setIR(uint8_t value) {
  return setPWM(CH_IR, value);
}

bool TLC59108Board::setLuxeon(uint8_t value) {
  return setPWM(CH_LUXEON, value);
}

bool TLC59108Board::allOff() {
  bool ok = true;

  ok &= applyMainLedModes(0x00, 0x00, 0x00);
  ok &= setPWM(CH_UV, 0);
  ok &= setPWM(CH_IR, 0);
  ok &= setPWM(CH_LUXEON, 0);

  return ok;
}

bool TLC59108Board::uvOnly(uint8_t value) {
  bool ok = true;

  ok &= setPWM(CH_UV, value);
  ok &= setPWM(CH_IR, 0);
  ok &= setPWM(CH_LUXEON, 0);
  ok &= applyMainLedModes(0x02, 0x00, 0x00);

  return ok;
}

bool TLC59108Board::irOnly(uint8_t value) {
  bool ok = true;

  ok &= setPWM(CH_UV, 0);
  ok &= setPWM(CH_IR, value);
  ok &= setPWM(CH_LUXEON, 0);
  ok &= applyMainLedModes(0x00, 0x02, 0x00);

  return ok;
}

bool TLC59108Board::luxeonOnly(uint8_t value) {
  bool ok = true;

  ok &= setPWM(CH_UV, 0);
  ok &= setPWM(CH_IR, 0);
  ok &= setPWM(CH_LUXEON, value);
  ok &= applyMainLedModes(0x00, 0x00, 0x02);

  return ok;
}

bool TLC59108Board::setChannel(uint8_t ch, bool on, uint8_t pwm) {
  if (ch > 2) return false;

  _chPWM[ch] = pwm;
  _chMode[ch] = on ? 0x02 : 0x00;

  if (!setPWM(ch, on ? pwm : 0)) return false;
  return applyMainLedModes(_chMode[0], _chMode[1], _chMode[2]);
}

void TLC59108Board::testSequence(uint16_t delayMs) {
  allOff();
  delay(300);

  uvOnly(255);
  delay(delayMs);
  allOff();
  delay(300);

  irOnly(255);
  delay(delayMs);
  allOff();
  delay(300);

  luxeonOnly(255);
  delay(delayMs);
  allOff();
  delay(300);
}
