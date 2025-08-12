import pyautogui
import time
from vl53l8ch_yaml_utils import update_log_settings, update_cnh_bin_settings

def wait_for_image(images, confidence=0.8, image_timeout=5):
    if isinstance(images, str):
        images = [images]

    start = time.time()
    best_confidence = 0.0

    while time.time() - start < timeout:
        for image in images:
            try:
                location = pyautogui.locateOnScreen(image, confidence=confidence)
                if location:
                    return location
            except pyautogui.ImageNotFoundException as e:
                msg = str(e)
                if "highest confidence" in msg:
                    try:
                        best_confidence = max(best_confidence, float(msg.split("highest confidence = ")[-1]))
                    except:
                        pass
        time.sleep(0.2)

    print(f"Failed to find any of: {images}.\nWaited for {timeout} seconds.\nHighest match confidence during search: {best_confidence:.3f}")
    return None


def data_logging_cycle(num_frames, image_timeout, logging_timeout):
    # Move the pointer to the top-left-ish corner of the screen (bug fix)
    pyautogui.moveTo(100, 100)

    # Locate and click the Start Logging button
    start_logging_location = wait_for_image(['start_logging_button_default.png', 'start_logging_button_changed.png'], confidence=0.8, image_timeout=image_timeout)
    pyautogui.click(pyautogui.center(start_logging_location))

    # Save the time when the sensor starts logging data to check if it times out later
    start_time = time.time()

    print("Start logging button clicked.")
    time.sleep(0.5)
    print("Waiting for data logging to finish...")

    # Give GUI time to start logging
    time.sleep(4.5)

    # Wait until the data logging is complete
    while True:
        elapsed = time.time() - start_time
        if elapsed > logging_timeout:
            print(f"Data logging could not be completed within {logging_timeout} seconds. Clicking stop logging button.")
            pyautogui.click()  # pointer is already on the stop button
            break

        try:
            match = pyautogui.locateOnScreen('zero_sec.png', confidence=0.99)
        except pyautogui.ImageNotFoundException:
            match = None

        if match:
            print("Data logging complete.")
            break

        time.sleep(0.5)


def vl53l8ch_gui_startup(num_locations=1):
    # Prompt user for number of bins, start bin, and sub sample
    while True:
        try:
            num_bins = int(input("Enter the number of bins (1-18): "))
        except ValueError:
            print("Invalid input. Using default of 18 bins")
            num_bins = 18

        try:
            start_bin = int(input("Enter the start bin (0-127): "))
        except ValueError:
            print("Invalid input. Using default start bin of 0.")
            start_bin = 0

        try:
            max_sub_sample = (128 - start_bin) // num_bins
            sub_sample = int(input(f"Enter the sub sample (1-{max_sub_sample}): "))
        except ValueError:
            print("Invalid input. Using default sub sample of 1.")
            sub_sample = 1

        BIN_WIDTH_MM = 37.535
        first_bin_mm = (start_bin + sub_sample / 2) * BIN_WIDTH_MM
        last_bin_mm = (start_bin + (num_bins - 1) * sub_sample + sub_sample / 2) * BIN_WIDTH_MM

        print(f"CNH bin width: {BIN_WIDTH_MM * sub_sample} mm")
        print(f"First bin center: {first_bin_mm:.2f} mm")
        print(f"Last bin center : {last_bin_mm:.2f} mm")

        confirm = input("Enter 'y' to confirm, or any other key to re-enter: ")
        if confirm == 'y':
            break

    preset_name = "CUSTOM_CNH_8x8"
    update_cnh_bin_settings(preset_name, start_bin, sub_sample, num_bins)

    while True:
        try:
            num_frames = int(input(f"Enter the number of frames to log at each of {num_locations} locations: "))
        except ValueError:
            print("Invalid input. Using default of 10 frames per location.")
            num_frames = 10

        confirm = input("Enter 'y' to confirm, or any other key to re-enter: ")
        if confirm == 'y':
            break

    update_log_settings(num_frames=num_frames)

    # Start ranging
    start_ranging_location = pyautogui.locateOnScreen('start_ranging_button.png', confidence=0.8)
    pyautogui.click(pyautogui.center(start_ranging_location))
    print("\nStart ranging button clicked.\n")
    time.sleep(5)

    return num_frames