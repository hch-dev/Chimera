import os
from PIL import Image
import numpy as np

# This path is where the mock or real screenshot is stored temporarily
SCREENSHOT_PATH = "temp_screenshot.png"

def capture_screenshot(url, path=SCREENSHOT_PATH):
    """
    MOCK IMPLEMENTATION: Creates a dummy screenshot image file 
    instead of using a real headless browser.

    :param url: The target URL (unused in mock).
    :param path: The path to save the screenshot file.
    :return: The filepath if successful, otherwise None.
    """
    from log import LOG
    
    LOG.info(f"Starting MOCK headless browser capture for: {url}")
    
    try:
        # Create a simple 128x128 green placeholder image using PIL
        img_array = np.zeros((128, 128, 3), dtype=np.uint8)
        img_array[:, :, 1] = 180  # Set green channel high
        
        img = Image.fromarray(img_array, 'RGB')
        img.save(path)
        
        LOG.debug(f"Mock screenshot successfully saved to: {path}")
        return path

    except Exception as e:
        # Note: If PIL is not installed, the error will happen here, 
        # which is why Step 1 is important.
        LOG.error(f"Error during MOCK screenshot creation: {e}")
        return None

def cleanup_screenshot(path=SCREENSHOT_PATH):
    """Removes the temporary screenshot file."""
    if os.path.exists(path):
        os.remove(path)
        
if __name__ == '__main__':
    # Simple test case for the utility
    test_url = "https://www.google.com"
    screenshot_file = capture_screenshot(test_url)
    if screenshot_file:
        print(f"Test mock screenshot created: {screenshot_file}")
        cleanup_screenshot(screenshot_file)
    else:
        print("Test failed.")