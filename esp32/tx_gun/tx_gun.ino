// ===== ESP32-C3 TX (ARMA) - PRODUÇÃO com DEBOUNCE do gatilho =====
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_idf_version.h>
#include <string.h>

#ifndef ESP_IDF_VERSION_MAJOR
#define ESP_IDF_VERSION_MAJOR 4
#endif

#define WIFI_CHANNEL   1
#define SERIAL_BAUD    115200
#define INVERT_INPUTS  1   // 1 => bit=1 significa "pressionado" (recomendado com INPUT_PULLUP)

// pinos (ajuste se precisar)
#define BTN_TRIGGER  10
#define BTN_A        7
#define BTN_B        21

// === Parâmetros de debounce do gatilho (ajuste fino) ===
#define TRIG_MIN_RETRIGGER_US   3000   // período refratário após PRESS (ignora re-cliques por X us)
#define TRIG_RELEASE_STABLE_US  1500   // tempo que o sinal precisa ficar "solto" para liberar de vez

// MAC do RX (PC) - COLOQUE O SEU
static const uint8_t RX_MAC[6] = {0xDC,0x06,0x75,0x69,0x41,0xAC};

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

static inline uint8_t readButtonsRaw() {
  int t = digitalRead(BTN_TRIGGER);
  int a = digitalRead(BTN_A);
  int b = digitalRead(BTN_B);
#if INVERT_INPUTS
  t = !t; a = !a; b = !b;  // com INPUT_PULLUP, pressionado vira 1
#endif
  uint8_t bs = 0;
  bs |= (t & 1) << 0;  // bit0 = trigger
  bs |= (a & 1) << 2;  // bit2 = A
  bs |= (b & 1) << 3;  // bit3 = B
  return bs;
}

uint16_t g_seq = 0;
uint8_t  last_bs_sent = 0xFF; // força primeiro envio

// --- estado do filtro do gatilho ---
static uint8_t  trig_out = 0;             // estado filtrado exposto (0 solto, 1 pressionado)
static uint32_t trig_rearm_at = 0;        // micros() a partir do qual permitimos novo PRESS
static uint32_t trig_rel_start = 0;       // início da janela de release estável (0 = none)

#if ESP_IDF_VERSION_MAJOR >= 5
void onSend(const uint8_t *mac, esp_now_send_status_t status) { /* opcional */ }
#else
void onSend(uint8_t *mac, esp_now_send_status_t status) { /* opcional */ }
#endif

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(50);

  pinMode(BTN_TRIGGER, INPUT_PULLUP);
  pinMode(BTN_A,       INPUT_PULLUP);
  pinMode(BTN_B,       INPUT_PULLUP);

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (esp_now_init() != ESP_OK) { while(1) delay(1000); }
  esp_now_register_send_cb(onSend);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, RX_MAC, 6);
  peer.channel = WIFI_CHANNEL;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  esp_now_add_peer(&peer);
}

void loop() {
  uint32_t now = micros();
  uint8_t  raw = readButtonsRaw();

  // ---- Debounce do gatilho (bit0 de raw) ----
  uint8_t t_now = (raw & 0x01) ? 1 : 0;

  if (t_now) {
    // Pressionado: libere imediatamente se não estiver em período refratário
    trig_rel_start = 0; // cancela qualquer release pendente
    if (!trig_out && (int32_t)(now - trig_rearm_at) >= 0) {
      trig_out = 1;
      trig_rearm_at = now + TRIG_MIN_RETRIGGER_US; // bloqueia re-cliques por X us
    }
  } else {
    // Solto: só libere após ficar estável por Y us
    if (trig_out) {
      if (trig_rel_start == 0) trig_rel_start = now; // começou janela estável
      if ((now - trig_rel_start) >= TRIG_RELEASE_STABLE_US) {
        trig_out = 0;
        trig_rel_start = 0;
      }
    } else {
      trig_rel_start = 0; // já está solto
    }
  }

  // ---- Monta estado filtrado final (só substitui bit do gatilho) ----
  uint8_t bs_filtered = (raw & (uint8_t)~0x01) | (trig_out ? 0x01 : 0x00);

  if (bs_filtered != last_bs_sent) {
    last_bs_sent = bs_filtered;

    ShootPkt pkt;
    pkt.seq = ++g_seq;
    pkt.t_us = now;           // timestamp do TX
    pkt.buttonState = bs_filtered;
    pkt.checksum = checksum8((uint8_t*)&pkt, sizeof(pkt)-1);

    esp_now_send(RX_MAC, (uint8_t*)&pkt, sizeof(pkt));
  }

  // polling rápido (2 kHz). Pode baixar para 200 us se quiser expor ainda menos bounce.
  delayMicroseconds(500);
}
