"""Versioned constants shared by the stdlib-only controller and Torch trainer."""

SKILLS = (
    "push",
    "circle_left",
    "circle_right",
    "sweep_left",
    "sweep_right",
    "brace_counter",
    "disengage",
    "pin",
    "recover",
)
FRAMES, HEIGHT, WIDTH = 4, 36, 64
PIXELS = FRAMES * 3 * HEIGHT * WIDTH
VECTOR = 45 + 4 * 96 + 2
CONTRACT = "go2-tactics-v3-rgb4x36x64-options9"
