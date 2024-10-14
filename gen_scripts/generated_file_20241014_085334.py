import bpy

# Clear the existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Function to create a torus resembling a donut
def create_donut(location=(0, 0, 0), major_radius=1, minor_radius=0.3):
    bpy.ops.mesh.primitive_torus_add(align='WORLD', location=location, major_radius=major_radius, minor_radius=minor_radius)
    return bpy.context.object

# Function to add a sun lamp
def add_sun_lamp(location=(10, -10, 10)):
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=location)
    return bpy.context.object

# Function to setup a camera
def create_camera(location=(0, -15, 0)):
    bpy.ops.object.camera_add(align='WORLD', location=location)
    camera = bpy.context.object
    # Point the camera at the origin
    bpy.ops.object.select_all(action='DESELECT')
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera

    direction = bpy.context.scene.cursor.location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

# Function to setup rendering and save the image
def setup_render(file_path='./gen_images/render.png'):
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