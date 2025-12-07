try:
    import pyray as pr
    print("Imported pyray")
    pr.init_window(100, 100, "Test")
    pr.close_window()
except ImportError:
    print("pyray failed")

try:
    import raylib
    print("Imported raylib (raw binding?)")
except ImportError:
    print("raylib failed")
