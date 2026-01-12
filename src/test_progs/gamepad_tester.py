def dump_gamepad_layout(js):
    import pygame
    pygame.event.pump()
    print("name:", js.get_name())
    print("axes:", js.get_numaxes(), "buttons:", js.get_numbuttons(), "hats:", js.get_numhats())
    for i in range(js.get_numaxes()):
        print("axis", i, "=", js.get_axis(i))
    for i in range(js.get_numhats()):
        print("hat", i, "=", js.get_hat(i))

dump_gamepad_layout(js)
