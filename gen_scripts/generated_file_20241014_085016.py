import bpy
import os

# Clear the current scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Function to create a torus (donut)
def create_donut(location=(0, 0, 0), major_radius=1, minor_radius=0.3):
    bpy.ops.mesh.primitive_torus_add(align='WORLD', location=location, major_radius=major_radius, minor_radius=minor_radius)
    return bpy.context.object

# Create a donut at the origin
donut = create_donut()

# Function to add a sun lamp to the scene
def create_sun_light(location=(5, 5, 5)):
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=location)
    sun = bpy.context.object
    sun.data.energy = 3  # Adjust the strength of the sun
    return sun

# Add a sun lamp to the scene
sun_light = create_sun_light()

# Function to set up the camera
def setup_camera(location=(0, -15, 5), look_at=(0, 0, 0)):
    bpy.ops.object.camera_add(align='WORLD', location=location)
    camera = bpy.context.object

    # Point camera at the donut
    direction = bpy.mathutils.Vector(look_at) - bpy.mathutils.Vector(camera.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    return camera

# Set up the camera 15 meters away with slight elevation
camera = setup_camera()

# Set the render settings
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = os.path.join('./gen_images', 'donut_the_last.png')

# Render the scene
bpy.ops.render.render(write_still=True)