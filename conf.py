from typing import Literal
import os



######## AI 
PROVIDERS = Literal['AKKODIS', 'OPENAI']

AKKODIS_API_KEY: str = os.getenv("AKKODIS_API_KEY", "") # primary
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "") # secondary
PROVIDER: PROVIDERS = os.getenv("PROVIDER", "AKKODIS")

if AKKODIS_API_KEY == "" and PROVIDER == 'AKKODIS':
    raise Exception("You've selected AKKODIS API for accessing the models without providing your api key.")


if OPENAI_API_KEY == "" and PROVIDER == 'OPENAI':
    raise Exception("You've selected OPENAI API for accessing the models without providing your api key.")

if PROVIDER not in ['AKKODIS', 'OPENAI']:
    raise Exception("You have not selected a valid provider.")
#########



####### BLENDER
BLENDER_PATH: str = os.getenv("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")

####### generated 
GEN_PATH: str = os.getenv("GEN_PATH", "./gen_scripts/")
GEN_IMG_PATH: str = os.getenv("GEN_IMG_PATH", "./gen_images/")

####### LOG PATH
LOG_PATH: str = os.getenv("LOG_PATH",'App.log')
EMBED_PATH: str = os.getenv("EMBED_PATH",'v_blender_api_embeds.npy')
META_PATH: str = os.getenv("META_PATH",'blender_api_metadata.json')
MANUAL_EMBED_PATH: str = os.getenv("MANUAL_EMBED_PATH",'v_blender_manual_embeds.npy')
MANUAL_META_PATH: str = os.getenv("MANUAL_META_PATH",'blender_manual_metadata.json')


