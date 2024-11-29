/*
    Video: https://www.youtube.com/watch?v=oCMOYS71NIU
    Based on Neil Kolban example for IDF: https://github.com/nkolban/esp32-snippets/blob/master/cpp_utils/tests/BLE%20Tests/SampleNotify.cpp
    Ported to Arduino ESP32 by Evandro Copercini
    updated by chegewara

   Create a BLE server that, once we receive a connection, will send periodic notifications.
   The service advertises itself as: 4fafc201-1fb5-459e-8fcc-c5c9c331914b
   And has a characteristic of: beb5483e-36e1-4688-b7f5-ea07361b26a8

   The design of creating the BLE server is:
   1. Create a BLE Server
   2. Create a BLE Service
   3. Create a BLE Characteristic on the Service
   4. Create a BLE Descriptor on the characteristic
   5. Start the service.
   6. Start advertising.

   A connect handler associated with the server starts a background task that performs notification
   every couple of seconds.
*/
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLE2901.h>

BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
BLE2901 *descriptor_2901 = NULL;

bool deviceConnected = false;
bool oldDeviceConnected = false;
uint32_t value = 0;

#define BTN_TRIGGER 10
#define BTN_A 7
#define BTN_B 21

#define SOLENOID 2

int lastButtonState = 0;


int triggerStartPress = 0;
int triggerTimePressed = 0;
int triggerTimeReleased = 0;
int triggerTimeSinceRelease = 0;
int triggerLastState = HIGH;

// See the following for generating UUIDs:
// https://www.uuidgenerator.net/

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer, esp_ble_gatts_cb_param_t *param) {
    deviceConnected = true;
    esp_bd_addr_t remote_bda;
    memcpy(remote_bda, param->connect.remote_bda, sizeof(esp_bd_addr_t));

    // Request a connection parameter update
    pServer->updateConnParams(remote_bda, 6, 6, 0, 100);
  };

  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
  }
};

void setup() {

  pinMode(BTN_TRIGGER, INPUT_PULLUP);
  pinMode(BTN_A, INPUT_PULLUP);
  pinMode(BTN_B, INPUT_PULLUP);

  pinMode(SOLENOID, OUTPUT);
  
  Serial.begin(115200);

  // Create the BLE Device
  BLEDevice::init("ESP32");

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create a BLE Characteristic
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_INDICATE
  );

  // Creates BLE Descriptor 0x2902: Client Characteristic Configuration Descriptor (CCCD)
  pCharacteristic->addDescriptor(new BLE2902());
  // Adds also the Characteristic User Description - 0x2901 descriptor
  descriptor_2901 = new BLE2901();
  descriptor_2901->setDescription("My own description for this characteristic.");
  descriptor_2901->setAccessPermissions(ESP_GATT_PERM_READ);  // enforce read only - default is Read|Write
  pCharacteristic->addDescriptor(descriptor_2901);

  // Start the service
  pService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);  // set value to 0x00 to not advertise this parameter
  BLEDevice::startAdvertising();
  Serial.println("Waiting a client connection to notify...");
}

void loop() {
  // notify changed value

  int buttonState = 0;
  int triggerState = digitalRead(BTN_TRIGGER);
  buttonState |= triggerState;
  buttonState |= digitalRead(BTN_A) << 2;
  buttonState |= digitalRead(BTN_B) << 3;

  if (triggerState == LOW && triggerLastState == HIGH)
     {
      triggerStartPress = millis();
      digitalWrite(SOLENOID, HIGH);
      triggerLastState = LOW;
      Serial.println("Pressed");
      Serial.print(triggerStartPress);
    }

    if (triggerState == HIGH && triggerLastState == LOW) 
    {
    triggerLastState = HIGH;
    triggerStartPress = 0;
    triggerTimeReleased = millis();
    Serial.println("Released");
    digitalWrite(SOLENOID, LOW);
  }

  if (deviceConnected) 
  {
    if (buttonState != lastButtonState)
    {
      lastButtonState = buttonState;
      pCharacteristic->setValue((uint8_t *)&buttonState, 4);
      pCharacteristic->notify();
      Serial.println("Sending notification...");
    }
  }

    



  if (triggerState == LOW) {
    triggerTimePressed = millis() - triggerStartPress;

    if (triggerTimePressed >= 25 && triggerTimePressed <= 300)
    {
      digitalWrite(SOLENOID, LOW);
    }

    if (triggerTimePressed > 300) 
    {
      int cycle_pos = triggerTimePressed % 125;
      if (cycle_pos >= 50 && cycle_pos < 75)
      {
        digitalWrite(SOLENOID, HIGH);
      }
      else
      {
        digitalWrite(SOLENOID, LOW);
      }
    }
  }

  // disconnecting
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);                   // give the bluetooth stack the chance to get things ready
    pServer->startAdvertising();  // restart advertising
    Serial.println("start advertising");
    oldDeviceConnected = deviceConnected;
  }
  // connecting
  if (deviceConnected && !oldDeviceConnected) {
    // do stuff here on connecting
    oldDeviceConnected = deviceConnected;
  }

  delay(1);
}
