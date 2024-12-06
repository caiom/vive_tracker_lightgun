#include <AbsMouse.h>

byte buffer[9];

float x;
float y;
uint16_t move_x;
uint16_t move_y;
uint8_t mouse_pressed;
uint8_t mouse_pressed2;
uint8_t code;
bool pressed_left = false;
bool pressed_right = false;
bool pressed_middle = false;

void setup() {
 Serial.begin(115200);
 Serial.setTimeout(1);
 AbsMouse.init();
}
void loop() 
{
  while (!Serial.available());
  Serial.readBytes(buffer, 11);
  code = buffer[0];
  if (code == 37)
  {
    x = *((float*)&buffer[1]);
    y = *((float*)&buffer[5]);
    mouse_pressed = buffer[9];
    mouse_pressed2 = buffer[10];
    move_x = (uint16_t) (x*32767.0);
    move_y = (uint16_t) (y*32767.0);
    AbsMouse.move(move_x, move_y);

    // Serial.println("New data");
    // Serial.println(x);
    // Serial.println(y);
    // Serial.println(move_x);
    // Serial.println(move_y);

    if (mouse_pressed != mouse_pressed2)
      return;

    if ((mouse_pressed & MOUSE_LEFT) == MOUSE_LEFT)
    {
      if (!pressed_left)
      {
        AbsMouse.press(MOUSE_LEFT);
        pressed_left = true;
      }

    }
    else
    {
      if (pressed_left)
      {
        AbsMouse.release(MOUSE_LEFT);
        pressed_left = false;
      }
    }


    if ((mouse_pressed & MOUSE_RIGHT) == MOUSE_RIGHT)
    {
      if (!pressed_right)
      {
        AbsMouse.press(MOUSE_RIGHT);
        pressed_right = true;
      }

    }
    else
    {
      if (pressed_right)
      {
        AbsMouse.release(MOUSE_RIGHT);
        pressed_right = false;
      }
    }


    if ((mouse_pressed & MOUSE_MIDDLE) == MOUSE_MIDDLE)
    {
      if (!pressed_middle)
      {
        AbsMouse.press(MOUSE_MIDDLE);
        pressed_middle = true;
      }

    }
    else
    {
      if (pressed_middle)
      {
        AbsMouse.release(MOUSE_MIDDLE);
        pressed_middle = false;
      }
    }





  }
}