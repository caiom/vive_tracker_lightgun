// ===== ESP32-C3 RX (PC) - PRODUÇÃO =====
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_idf_version.h>

#ifndef ESP_IDF_VERSION_MAJOR
#define ESP_IDF_VERSION_MAJOR 4
#endif

#define WIFI_CHANNEL 1
#define SERIAL_BAUD 1000000

typedef struct __attribute__((packed)) {
  uint16_t seq;
  uint32_t t_us;
  uint8_t  buttonState;
  uint8_t  checksum;
} ShootPkt;

static inline uint8_t checksum8(const uint8_t* p, size_t n){
  uint32_t s=0; for(size_t i=0;i<n;i++) s+=p[i];
  return (uint8_t)(0x100 - (s & 0xFF));
}

volatile uint16_t last_seq = 0;

#if ESP_IDF_VERSION_MAJOR >= 5
void onRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
#else
void onRecv(const uint8_t *mac, const uint8_t *data, int len) {
#endif
  if (len != (int)sizeof(ShootPkt)) return;
  ShootPkt pkt; memcpy(&pkt, data, sizeof(pkt));
  uint8_t cs = checksum8((uint8_t*)&pkt, sizeof(pkt)-1);
  if (cs != pkt.checksum) return;

  if ((uint16_t)(pkt.seq - last_seq) == 0) return;
  if ((uint16_t)(pkt.seq - last_seq) < 0x8000) last_seq = pkt.seq;

  const uint8_t hdr[2] = {0xAA, 0x55};
  Serial.write(hdr, 2);
  Serial.write((uint8_t*)&pkt, sizeof(pkt));
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(50);

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (esp_now_init() != ESP_OK) {
    while(1) delay(1000);
  }
  esp_now_register_recv_cb(onRecv);
}

void loop() { delay(1); }
