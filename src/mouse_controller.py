'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
'''
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':0.7, 'slow':10, 'medium':5}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, x, y):
        pyautogui.moveRel(-1*x*self.precision, -1*y*self.precision, duration=self.speed)

    # def move_to_center(self):
    #     x, y = pyautogui.size()
    #     if pyautogui.onScreen(x, y):
    #         pyautogui.moveTo(-0.5*x, -0.5*y, duration=self.speed)
