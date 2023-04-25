

int startPress = 0;
int timePressed = 0;
int timeReleased = 0;
int timeSinceRelease = 0;
int lastState = HIGH;

void setup() {
  // put your setup code here, to run once:
  pinMode(9, OUTPUT);
  pinMode(PIND5, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop() {

  int triggerState = digitalRead(PIND5);

  if (triggerState == LOW && lastState == HIGH) {
    startPress = millis();
    digitalWrite(9, HIGH);
    delay(25);
    digitalWrite(9, LOW);
    lastState = LOW;
    Serial.println("Pressed");
    Serial.print(startPress);
  }

  if (triggerState == HIGH && lastState == LOW) {
    lastState = HIGH;
    startPress = 0;
    timeReleased = millis();
  }

  if (triggerState == LOW) {
    timePressed = millis() - startPress;

    if (timePressed > 300) {
      Serial.println("Machine GUN!");
      Serial.println(timePressed);
      digitalWrite(9, HIGH);
      delay(25);
      digitalWrite(9, LOW);
      delay(100);
    }
  }

  delay(2);
}
