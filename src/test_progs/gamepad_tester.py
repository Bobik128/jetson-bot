import pygame

def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Gamepad connected: {js.get_name()}")
    return js

def dump_gamepad_layout(js):
    pygame.event.pump()
    print("name:", js.get_name())
    print("axes:", js.get_numaxes(), "buttons:", js.get_numbuttons(), "hats:", js.get_numhats())
    for i in range(js.get_numaxes()):
        print("axis", i, "=", js.get_axis(i))
    for i in range(js.get_numhats()):
        print("hat", i, "=", js.get_hat(i))

dump_gamepad_layout(init_gamepad())
