 #include <Mouse.h>

byte buffer[9];

int x;
int y;
uint8_t mouse_buttom;

void setup() {
 Serial.begin(115200);
 Serial.setTimeout(1);
 Mouse.begin();
}
void loop() 
{
  while (!Serial.available());
  Serial.readBytes(buffer, 9);
  x = *((int*)&buffer[0]);
  y = *((int*)&buffer[4]);
  mouse_buttom = buffer[8];
  Mouse.move(x, y);

  if (mouse_buttom > 0)
    Mouse.click(mouse_buttom);
}