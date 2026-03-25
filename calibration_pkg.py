import cv2
import numpy as np
import os
import glob
import shutil

def create_directory(directory_name):
    # Check if directory exists
    if os.path.exists(directory_name):
        # Empty the directory if it exists
        print(f"Emptying existing '{directory_name}' directory...")
    
        for filename in os.listdir(directory_name):
            file_path = os.path.join(directory_name, filename)
            # if it is a file remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # if it is a directory recursively remove it
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    # If the directory doesnt exist yet create it
    else:
        print(f"Creating '{directory_name}' directory...")
        os.makedirs(directory_name)

def extract_frames(frame_dir = "images", rotate_90_ccw = False, frame_interval = 40, video_name = "VID_20260323_131653.mp4"):
    # Create directory to store extracted frames
    create_directory(frame_dir)

    cap = cv2.VideoCapture(video_name)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_name}.")
        return

    frame_count = 0
    saved_count = 0

    print(f"Processing '{video_name}'...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            # Rotate 90 degrees counter-clockwise if enabled
            if rotate_90_ccw:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            file_path = os.path.join(frame_dir, f"image_{saved_count}.jpg")
            cv2.imwrite(file_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count - 1} frames.")

def get_calibration(calibration_dir = "calibration_images", CHECKERBOARD = (10,7), image_directory = './images', square_size=25):
    # Create directory to store calibration images
    create_directory(calibration_dir)
    print(f"Calibrating with extracted frames...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    saved_count = 0

    # Extracting path of individual image stored in a given directory
    images = glob.glob(f'{image_directory}/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
            # This draws a small red circle on the very first corner detected
            cv2.circle(img, (int(corners2[0][0][0]), int(corners2[0][0][1])), 10, (0, 0, 255), -1)
        
        file_path = os.path.join(calibration_dir, f"calibration_image_{saved_count}.jpg")
        cv2.imwrite(file_path, img)
        saved_count += 1

    cv2.destroyAllWindows()

    h,w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return mtx, dist, objp

def get_world_coords(mtx, dist, objp, CHECKERBOARD=(10, 7), image_directory = './images'):
    """ Uses solvePnP to find the rotation and translation of a specific image """
    
    # Extracting path of individual image stored in a given directory
    images = glob.glob(f'{image_directory}/*.jpg')

    world_coords = []

    for img_name in images:
        img = cv2.imread(img_name)
        assert img is not None, "Image not found!"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            # solvePnP finds the pose of the object (world) relative to the camera
            success, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
            
            if success:
                print(f"\nResults for {img_name}:")
                print("Rotation Vector (rvec):\n", rvec)
                print("Translation Vector (tvec - distance from camera):\n", tvec)
                
                # Convert rotation vector to matrix for coordinate math
                rmat, _ = cv2.Rodrigues(rvec)
                
                # The position of the camera in world coordinates
                cam_pos = -np.matrix(rmat).T * np.matrix(tvec)
                print("Camera Position in World Coords:\n", cam_pos)
                world_coords.append(cam_pos)
        else:
            print(f"Could not find checkerboard in {image_directory}/{img_name}")
    return world_coords

if __name__ == "__main__":
    # Extract calibration images from video
    extract_frames(frame_dir = "images", rotate_90_ccw = False, frame_interval = 40, video_name = "VID_20260323_131653.mp4")
    # Get camera intrinsics (matrix and distortion)
    mtx, dist, objp_template = get_calibration(calibration_dir = "calibration_images", CHECKERBOARD = (10,7), image_directory = './images')
    # Use solvePnP on a specific frame to get extrinsics (world coords)
    get_world_coords(mtx, dist, objp_template, image_directory = './images')