#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
from shutil import which

'''
Usage Instructions
Basic Usage (Background Mode Default)

python run_blender_script.py <blender_script.py> [--blender-executable <path_to_blender>] [--gui] [-- <script_args>]

    <blender_script.py>: The Blender Python script you want to run.
    --blender-executable <path_to_blender>: Optional path to the Blender executable if it's not in your system PATH.
    --gui: Optional flag to run Blender with the GUI (since background mode is default).
    -- <script_args>: Optional arguments to pass to the Blender script after the -- separator.

Examples

Example 1: Run a Blender script in background mode (default behavior).

python run_blender_script.py my_blender_script.py -- output.png

Example 2: Run a Blender script with the GUI.

python run_blender_script.py my_blender_script.py --gui -- arg1 arg2

Example 3: Specify a custom Blender executable and run in background mode.

python run_blender_script.py my_blender_script.py --blender-executable "/path/to/blender" -- output.png

Additional Tips

Making the Script Executable:

If you're on a Unix-like system, you can make run_blender_script.py executable:


chmod +x run_blender_script.py

Then you can run it directly:

./run_blender_script.py my_blender_script.py -- output.png

Specifying the Blender Executable:

If Blender is not in your system PATH, or you want to use a specific version, use the --blender-executable option:

python run_blender_script.py my_blender_script.py --blender-executable "/path/to/blender" -- output.png

Passing Multiple Arguments:

You can pass multiple arguments to your Blender script after the -- separator:

python run_blender_script.py my_blender_script.py -- arg1 arg2 arg3

'''





def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run a Blender Python script from the command line.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'blender_script',
        help='Path to the Blender Python script to execute.'
    )

    # Optional arguments
    parser.add_argument(
        '--blender-executable',
        default='blender',
        help='Path to the Blender executable.'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Run Blender with GUI (default is background mode).'
    )

    # Arguments to pass to the Blender script
    parser.add_argument(
        'script_args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the Blender script (after "--").'
    )

    args = parser.parse_args()

    # Verify that the Blender script exists and is a file
    blender_script_path = os.path.abspath(args.blender_script)
    if not os.path.isfile(blender_script_path):
        print(f"Error: Blender script '{blender_script_path}' does not exist or is not a file.")
        sys.exit(1)

    # Resolve the Blender executable path
    blender_executable = args.blender_executable
    if not os.path.isfile(blender_executable):
        blender_executable = which(blender_executable)
        if blender_executable is None:
            print(f"Error: Blender executable '{args.blender_executable}' not found in PATH.")
            sys.exit(1)

    # Construct the command to run Blender with your script
    command = [
        blender_executable
    ]

    if not args.gui:
        command.append('--background')

    command.extend([
        '--python', blender_script_path,
        '--'
    ])

    if args.script_args:
        command.extend(args.script_args)

    # Display the command for debugging purposes
    print("Executing command:")
    print(' '.join(f'"{c}"' if ' ' in c else c for c in command))

    # Run the command
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Blender script execution failed with return code {e.returncode}.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
