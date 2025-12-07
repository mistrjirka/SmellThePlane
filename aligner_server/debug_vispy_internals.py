import os
os.environ["VISPY_GL_LIB"] = "libGLESv2.so.2"

import vispy
vispy.use(gl='gl2')

from vispy import app, gloo
from vispy.gloo import gl
import ctypes

class Canvas(app.Canvas):
    def on_initialize(self, event):
        print("Initialize")
        try:
            print(f"gl2 module: {gl}")
            if hasattr(gl, '_lib'):
                lib = gl._lib
                # Define return type for glGetString as c_char_p
                lib.glGetString.restype = ctypes.c_char_p
                lib.glGetString.argtypes = [ctypes.c_uint]
                
                print(f"GL Version: {lib.glGetString(gl.GL_VERSION)}")
                print(f"GL Vendor: {lib.glGetString(gl.GL_VENDOR)}")
                print(f"GL Renderer: {lib.glGetString(gl.GL_RENDERER)}")
                print(f"GL Shading Language Version: {lib.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)}")
            else:
                 print("No _lib found on gl2 module")

        except Exception as e:
             print(f"FAILED to call glGetString: {e}")
             import traceback
             traceback.print_exc()

        # Check VisPy's internal capability flags
        from vispy.gloo.context import get_current_canvas
        canvas = get_current_canvas()
        if canvas and canvas.context:
             print(f"Context Config: {canvas.context.config}")
             print(f"Context Class: {type(canvas.context)}")
             # Inspect the configuration that determines shader version
             if hasattr(canvas.context, 'gl_version'):
                  print(f"Detected GL Version: {canvas.context.gl_version}")
        
        print(vispy.sys_info())
        app.quit()

c = Canvas(show=True, size=(100, 100))
app.run()
