// Include Libraries
#include "Arduino.h"
#include "Buzzer.h"
#include "LiquidCrystal.h"
#include "Adafruit_NeoPixel.h"
#include "Servo.h"
// #include "pitches.h"

// Pin Definitions
#define BUZZER_PIN_SIG  2
#define LCD_PIN_RS  8
#define LCD_PIN_E 7
#define LCD_PIN_DB4 3
#define LCD_PIN_DB5 4
#define LCD_PIN_DB6 5
#define LCD_PIN_DB7 6
#define LEDRGB_PIN_DIN  9
#define SERVO9G_PIN_SIG 10



// Global variables and defines
#define LedRGB_NUMOFLEDS 1
const int servo9gRestPosition   = 20;  //Starting position
const int servo9gTargetPosition = 150; //Position when event is detected
// object initialization
Buzzer buzzer(BUZZER_PIN_SIG);
LiquidCrystal lcd(LCD_PIN_RS,LCD_PIN_E,LCD_PIN_DB4,LCD_PIN_DB5,LCD_PIN_DB6,LCD_PIN_DB7);
Adafruit_NeoPixel LedRGB(LedRGB_NUMOFLEDS, LEDRGB_PIN_DIN, NEO_GRB + NEO_KHZ800);
Servo servo9g;


// define vars for testing menu
const int timeout = 10000;       //define timeout of 10 sec
char menuOption = 0;
long time0;

// Setup the essentials for your circuit to work. It runs first every time your circuit is powered with electricity.
void setup() 
{
    // Setup Serial which is useful for debugging
    // Use the Serial Monitor to view printed messages
    Serial.begin(9600);
    while (!Serial) ; // wait for serial port to connect. Needed for native USB
    Serial.println("start");
    
    // set up the LCD's number of columns and rows
    lcd.begin(16, 2);
    LedRGB.begin(); // This initializes the NeoPixel library.
    LedRGB.show(); // Initialize all leds to 'off'
    servo9g.attach(SERVO9G_PIN_SIG);
    servo9g.write(servo9gRestPosition);
    delay(100);
    servo9g.detach();
    menuOption = menu();
    
}

// Main logic of your circuit. It defines the interaction between the components you selected. After setup, it runs over and over again, in an eternal loop.
void loop() 
{
    
    
    if(menuOption == '1') {
    // Buzzer - Test Code
      buzzer.begin(100);
      buzzer.sound(NOTE_E7, 80);
      buzzer.sound(NOTE_E7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_E7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_C7, 80);
      buzzer.sound(NOTE_E7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_G7, 80);
      buzzer.sound(0, 240);
      buzzer.sound(NOTE_G6, 80);
      buzzer.sound(0, 240);
      buzzer.sound(NOTE_C7, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_G6, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_E6, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_A6, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_B6, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_AS6, 80);
      buzzer.sound(NOTE_A6, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_G6, 100);
      buzzer.sound(NOTE_E7, 100);
      buzzer.sound(NOTE_G7, 100);
      buzzer.sound(NOTE_A7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_F7, 80);
      buzzer.sound(NOTE_G7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_E7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_C7, 80);
      buzzer.sound(NOTE_D7, 80);
      buzzer.sound(NOTE_B6, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_C7, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_G6, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_E6, 80);
      buzzer.sound(0, 160);
      buzzer.sound(NOTE_A6, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_B6, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_AS6, 80);
      buzzer.sound(NOTE_A6, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_G6, 100);
      buzzer.sound(NOTE_E7, 100);
      buzzer.sound(NOTE_G7, 100);
      buzzer.sound(NOTE_A7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_F7, 80);
      buzzer.sound(NOTE_G7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_E7, 80);
      buzzer.sound(0, 80);
      buzzer.sound(NOTE_C7, 80);
      buzzer.sound(NOTE_D7, 80);
      buzzer.sound(NOTE_B6, 80);
      buzzer.sound(0, 160);

      buzzer.end(2000);
    }
    else if(menuOption == '2') {
    // LCD 16x2 - Test Code
    // Print a message to the LCD.
    lcd.setCursor(0, 0);
    lcd.print("Circuito Rocks !");
    // Turn off the display:
    lcd.noDisplay();
    delay(500);
    // Turn on the display:
    lcd.display();
    delay(500);
    }
    else if(menuOption == '3') {
    // 9G Micro Servo - Test Code
    // The servo will rotate to target position and back to resting position with an interval of 500 milliseconds (0.5 seconds) 
    servo9g.attach(SERVO9G_PIN_SIG);         // 1. attach the servo to correct pin to control it.
    servo9g.write(servo9gTargetPosition);  // 2. turns servo to target position. Modify target position by modifying the 'ServoTargetPosition' definition above.
    delay(500);                              // 3. waits 500 milliseconds (0.5 sec). change the value in the brackets (500) for a longer or shorter delay in milliseconds.
    servo9g.write(servo9gRestPosition);    // 4. turns servo back to rest position. Modify initial position by modifying the 'ServoRestPosition' definition above.
    delay(500);                              // 5. waits 500 milliseconds (0.5 sec). change the value in the brackets (500) for a longer or shorter delay in milliseconds.
    servo9g.detach();                    // 6. release the servo to conserve power. When detached the servo will NOT hold it's position under stress.
    }
    
    if (millis() - time0 > timeout)
    {
        menuOption = menu();
    }
    
}



// Menu function for selecting the components to be tested
// Follow serial monitor for instrcutions
char menu()
{

    Serial.println(F("\nWhich component would you like to test?"));
    Serial.println(F("(1) Buzzer"));
    Serial.println(F("(2) LCD 16x2"));
    Serial.println(F("(3) 9G Micro Servo"));
    Serial.println(F("(menu) send anything else or press on board reset button\n"));
    while (!Serial.available());

    // Read data from serial monitor if received
    while (Serial.available()) 
    {
        char c = Serial.read();
        if (isAlphaNumeric(c)) 
        {   
            
            if(c == '1') 
          Serial.println(F("Now Testing Buzzer"));
        else if(c == '2') 
          Serial.println(F("Now Testing LCD 16x2"));
        else if(c == '3') 
          Serial.println(F("Now Testing 9G Micro Servo"));
            else
            {
                Serial.println(F("illegal input!"));
                return 0;
            }
            time0 = millis();
            return c;
        }
    }
}
