#include <Servo.h>
#include <LiquidCrystal.h>
#include <Buzzer.h>

Servo servoMotor;
LiquidCrystal lcd(8, 7, 3, 4, 5, 6);
int buzzerPin = 2; // Define the pin for the buzzer
Buzzer buzzer(buzzerPin);
void setup() {
  Serial.begin(9600);
  servoMotor.attach(10);
  lcd.begin(16, 2);
  pinMode(buzzerPin, OUTPUT); // Set the buzzer pin as an output
}

// dictionary of known people
struct KeyValue {
  const char* key;
  int value;
};

KeyValue dictionary[] = {
  {"aryan", 2},
  {"tharun", 3},
  {"aryabhatta", 4}
};

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();

    // debugging
    Serial.print("Received Signal: ");
    Serial.println(signal);

    // no face detected. Silence.
    if(signal == '0') {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Shhhhh");
    }
    // alarm
    else if(signal == '1') {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Access Denied");

      // Sound an alarm
      for (int i = 0; i < 3; i++) {
        buzzer.begin(2000);
        delay(200);
        buzzer.end(2000);
        delay(200);
      }
    }
    else {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Welcome, ");
      lcd.print(dictionary[signal].key);

      // Sound a normal opening sound
      tone(buzzerPin, 1000, 500); // Play a tone of 1000 Hz for 500 ms
      
      // Rotate the servo motor to open the gate
      servoMotor.write(90);
      delay(3000);
      for (int i = 0; i < 3; i++) {
        buzzer.begin(1000);
        delay(100);
        buzzer.end(1000);
        delay(50);
      } 
      servoMotor.write(0); // Move servo back to initial position
    }
  }
}
