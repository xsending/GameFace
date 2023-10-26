# GameFace
Undergraduate Thesis project that revolves around Computer Vision, specifically MediaPipe and some Haar Cascade.

## Description
This project captures the face of the user using the webcam and relies on the changes of user's face and head to control the keyboard. Certain actions produce an equivalent input on the keyboard. 
The project also includes a mouse mode which allows the control of the mouse.
The project solely relies on the facial and head movements to produce keyboard and mouse inputs.
This project is mainly aimed towards disabled people, specifically those with disabilities on the arms or hands.

## Getting Started

### How to use the program
1. To execute the program, open the main.py file and run the code.
2. A new window will appear and will ask the user to perform an open mouth movement for its calibration.
3. The user's initial distance from the web camera will serve as the basis for the program to determine the proximity of the user's head from the camera.
4. Make sure to position yourself at a comfortable distance away from the web camera.
5. Once calibration is done, the application can be used.
6. Initial controls (for keyboard) are the following (Optimized for playing movement games):
   - Lean Forward = 'c'
   - Open mouth = 'x'
   - Smile = 'z'
   - Look Left = 'a'
   - Look Right = 'd'
   - Look Up  = 'w'
   - Look Down = 's'
   - Lean Back = Switch to mouse mode

   - If the user enters the mouse mode, a rectangular line guide will appear which represents the user's monitor dimension.
   - Mouse mode controls:
        - Head position = Mouse cursor position
        - Open mouth = left click
        - Smile = right click
        - Lean Back = Switch to keyboard mode

8. In order to change keyboard controls or perform other application functionalities, the user can proceed to the application's second tab.
9. Only the keyboard controls can be changed for the current version.
10. The user can switch to single press mode or multi press mode.
11. Single press mode registers a keyboard input only once for each head or face movement.
12. Multipress mode registers a keyboard input repeatedly as long as the user is performing the face or head movement.
13. Re-calibration can be used to re-calibrate the user's registered distance from the web camera.
14. Re-calibration can be used to turn off the application's detection of the user face and head movements.
15. An on-screen keyboard can be accessed to help in changing keyboard controls.

## Help
1. The movement detection may sometimes be buggy and will not be able to detect certain movements.
   Possible solution: Please perform a re-calibration

## Authors

Abel Gomez  
Kenneth Lim
Joshua Azarcon
John Dominic Contreras

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
