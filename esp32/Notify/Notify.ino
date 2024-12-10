#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLE2901.h>

// BLE objects and variables
BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
BLE2901 *descriptor_2901 = NULL;

// Connection tracking
bool deviceConnected = false;
bool oldDeviceConnected = false;

// Solenoid and button states
#define BTN_TRIGGER 10
#define BTN_A 7
#define BTN_B 21
#define SOLENOID 2

int lastButtonState = 0;

// Trigger button press timing
int triggerStartPress = 0;      // When the trigger was first pressed
int triggerTimePressed = 0;     // How long the trigger has been held down
int triggerLastState = HIGH;    // Last read state of the trigger button
bool solenoidActive = false;    // Whether the solenoid functionality is active

// General button states
int buttonState = 0;            // Composed bitfield of all button states
int triggerState = 0;
int buttonBState = 0;
int buttonBStartPress = 0;
int buttonBTimePressed = 0;
int buttonBLastState = HIGH;

unsigned long lastNotification = 0;  // Last time we sent a BLE notification

// BLE UUIDs
// https://www.uuidgenerator.net/
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"


// Callback class to handle BLE server events
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer, esp_ble_gatts_cb_param_t *param) {
    deviceConnected = true;

    // Copy the remote device address
    esp_bd_addr_t remote_bda;
    memcpy(remote_bda, param->connect.remote_bda, sizeof(esp_bd_addr_t));

    // Request a connection parameter update to reduce latency/power consumption
    pServer->updateConnParams(remote_bda, 6, 6, 0, 100);
  }

  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
  }
};

void setup() {
  // Set pin modes
  pinMode(BTN_TRIGGER, INPUT_PULLUP);
  pinMode(BTN_A, INPUT_PULLUP);
  pinMode(BTN_B, INPUT_PULLUP);
  pinMode(SOLENOID, OUTPUT);
  digitalWrite(SOLENOID, LOW);

  Serial.begin(115200);

  // Initialize the BLE Device with a given name
  BLEDevice::init("ESP32");

  // Create the BLE Server and set its callbacks
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create a BLE Characteristic with read, write, notify and indicate properties
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE |
    BLECharacteristic::PROPERTY_NOTIFY |
    BLECharacteristic::PROPERTY_INDICATE
  );

  // Add a Client Characteristic Configuration Descriptor (CCCD) to allow clients to enable/disable notifications
  pCharacteristic->addDescriptor(new BLE2902());

  // Add a Characteristic User Description (0x2901) descriptor
  descriptor_2901 = new BLE2901();
  descriptor_2901->setDescription("My own description for this characteristic.");
  descriptor_2901->setAccessPermissions(ESP_GATT_PERM_READ);  
  pCharacteristic->addDescriptor(descriptor_2901);

  // Start the service
  pService->start();

  // Start advertising the service
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);  // Not advertising this parameter
  BLEDevice::startAdvertising();

  Serial.println("Waiting for a client to connect...");
}

void loop() {
  // Read current input states
  triggerState = digitalRead(BTN_TRIGGER);
  int buttonAState = digitalRead(BTN_A);
  buttonBState = digitalRead(BTN_B); // <- Make sure the semicolon is here

  // Compose buttonState:
  // bit 0: triggerState
  // bit 2: buttonAState
  // bit 3: buttonBState
  // The other bits are 0 for now.
  buttonState = 0;
  buttonState |= (triggerState & 0x01);        // puts triggerState in bit 0
  buttonState |= (buttonAState & 0x01) << 2;   // puts buttonAState in bit 2
  buttonState |= (buttonBState & 0x01) << 3;   // puts buttonBState in bit 3

  // Handle trigger button press/release logic and solenoid control
  if (triggerState == LOW && triggerLastState == HIGH && solenoidActive) {
    // Trigger just pressed
    triggerStartPress = millis();
    digitalWrite(SOLENOID, HIGH);
    triggerLastState = LOW;
    Serial.println("Trigger pressed");
  }

  if (triggerState == HIGH && triggerLastState == LOW && solenoidActive) {
    // Trigger just released
    triggerLastState = HIGH;
    triggerStartPress = 0;
    digitalWrite(SOLENOID, LOW);
    Serial.println("Trigger released");
  }

  // Handle long-press on button B to toggle solenoidActive
  if (buttonBState == LOW && buttonBLastState == HIGH) {
    // Button B just pressed
    buttonBStartPress = millis();
    buttonBLastState = LOW;
  }

  if (buttonBState == HIGH && buttonBLastState == LOW) {
    // Button B just released
    buttonBLastState = HIGH;
    buttonBStartPress = 0;
  }

  // If Button B is held for more than 3 seconds, toggle solenoidActive
  if (buttonBState == LOW && ((millis() - buttonBStartPress) > 3000)) {
    buttonBStartPress = millis();
    solenoidActive = !solenoidActive;
  }

  // If a client is connected, send notifications when state changes or periodically
  if (deviceConnected) {
    // Notify on any change in button state or every 200ms
    if (buttonState != lastButtonState || ((millis() - lastNotification) > 200)) {
      lastButtonState = buttonState;
      // Send the buttonState as a 4-byte value
      pCharacteristic->setValue((uint8_t *)&buttonState, 4);
      pCharacteristic->notify();
      Serial.println("Sending notification...");
      lastNotification = millis();
    }
  }

  // If trigger is held down and solenoid is active, handle timing-based solenoid pulses
  if (triggerState == LOW && solenoidActive) {
    triggerTimePressed = millis() - triggerStartPress;

    // If held for 25-300ms, keep solenoid off
    if (triggerTimePressed >= 25 && triggerTimePressed <= 300) {
      digitalWrite(SOLENOID, LOW);
    }

    // After 300ms, pulse the solenoid in a pattern
    if (triggerTimePressed > 300) {
      int cycle_pos = triggerTimePressed % 125; // Cycle every 125ms
      if (cycle_pos >= 50 && cycle_pos < 75) {
        digitalWrite(SOLENOID, HIGH);
      } else {
        digitalWrite(SOLENOID, LOW);
      }
    }
  }

  // Handle disconnect scenario
  if (!deviceConnected && oldDeviceConnected) {
    // Give the stack time and then start advertising again
    delay(500);
    pServer->startAdvertising();
    Serial.println("Starting advertising again...");
    oldDeviceConnected = deviceConnected;
  }

  // Handle connect scenario
  if (deviceConnected && !oldDeviceConnected) {
    // If needed, do something special upon connection
    oldDeviceConnected = deviceConnected;
  }

  // A small delay to reduce busy-waiting
  delay(1);
}
