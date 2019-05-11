import bpy
from random import randint

# Create cube
# bpy.ops.mesh.primitive_cube_add()

model_path = "/home/enric/Documents/repos/3Dhackathon/models/sample_model/house.stl"


#import model
bpy.ops.import_mesh.stl(filepath=model_path)

# adjust camera
bpy.ops.view3d.camera_to_view_selected()

# Render it
bpy.data.scenes['Scene'].render.filepath = "render_model.png"
bpy.ops.render.render( write_still=True ) 
