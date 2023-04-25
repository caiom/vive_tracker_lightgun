 #include <Mouse.h>

byte buffer[9];

int x;
int y;
uint8_t mouse_pressed;
uint8_t mouse_pressed2;
uint8_t code;
bool pressed_left = false;
bool pressed_right = false;
bool pressed_middle = false;

void setup() {
 Serial.begin(115200);
 Serial.setTimeout(1);
 Mouse.begin();
}
void loop() 
{
  while (!Serial.available());
  Serial.readBytes(buffer, 11);
  code = buffer[0];
  if (code == 37)
  {
    x = *((int*)&buffer[1]);
    y = *((int*)&buffer[5]);
    mouse_pressed = buffer[9];
    mouse_pressed2 = buffer[10];
    Mouse.move(x, y);

    if (mouse_pressed != mouse_pressed2)
      return;

    if ((mouse_pressed & MOUSE_LEFT) == MOUSE_LEFT)
    {
      if (!pressed_left)
      {
        Mouse.press(MOUSE_LEFT);
        pressed_left = true;
      }

    }
    else
    {
      if (pressed_left)
      {
        Mouse.release(MOUSE_LEFT);
        pressed_left = false;
      }
    }


    if ((mouse_pressed & MOUSE_RIGHT) == MOUSE_RIGHT)
    {
      if (!pressed_right)
      {
        Mouse.press(MOUSE_RIGHT);
        pressed_right = true;
      }

    }
    else
    {
      if (pressed_right)
      {
        Mouse.release(MOUSE_RIGHT);
        pressed_right = false;
      }
    }


    if ((mouse_pressed & MOUSE_MIDDLE) == MOUSE_MIDDLE)
    {
      if (!pressed_middle)
      {
        Mouse.press(MOUSE_MIDDLE);
        pressed_middle = true;
      }

    }
    else
    {
      if (pressed_middle)
      {
        Mouse.release(MOUSE_MIDDLE);
        pressed_middle = false;
      }
    }





  }
}