import bpy

# Clear the existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Function to create a torus resembling a donut
def create_donut(location=(0, 0, 0), major_radius=1, minor_radius=0.3):
    bpy.ops.mesh.primitive_torus_add(align='WORLD', location=location, major_radius=major_radius, minor_radius=minor_radius)
    donut = bpy.context.object
    # Rotate the donut 30 degrees around the X-axis
    donut.rotation_euler[0] = 30 * (3.14159 / 180)  # Convert degrees to radians
    return donut

# Function to add a sun lamp and make it brighter
def add_sun_lamp(location=(10, -10, 10), energy=5):
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=location)
    sun_lamp = bpy.context.object
    sun_lamp.data.energy = energy  # Increase the brightness of the sun
    return sun_lamp

# Function to setup a camera
def create_camera(location=(15, 15, 0)):
    bpy.ops.object.camera_add(align='WORLD', location=location)
    camera = bpy.context.object
    # Point the camera at the origin
    bpy.ops.object.select_all(action='DESELECT')
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera

    target_location = (0, 0, 0)
    direction = bpy.context.scene.cursor.location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

# Function to setup rendering and save the image
def setup_render(file_path='./gen_images/donut_the_last.png'):
    bpy.context.scene.camera = bpy.context.scene.objects['Camera']
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = file_path
    bpy.ops.render.render(write_still=True)

# Create the donut, lamp, and camera
donut = create_donut()
sun_lamp = add_sun_lamp()
camera = create_camera()

# Setup render and save the image
setup_render()