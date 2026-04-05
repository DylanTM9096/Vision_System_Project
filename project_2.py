import cv2
import numpy as np
import os
import glob
import shutil
import tqdm
import matplotlib.pyplot as plt

def create_directory(directory_name):
    """Function to create an empty directory if one doesn't already exist with the same name.
    Otherwise the existing directory will be emptied.

    Parameters:
        directory_name: The name of the directory to create or empty

    Returns:
        True if the directory now exists and is empty, otherwise False
    """
    success = True
    # Check if directory exists
    if os.path.exists(directory_name):
        # Empty the directory if it exists
        print(f"Emptying existing '{directory_name}' directory...")

        files = os.listdir(directory_name)
    
        for filename in tqdm.tqdm(files, desc="Deleting files", unit="file"):
            file_path = os.path.join(directory_name, filename)
            try:
                # if it is a file or shortcut remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # if it is a directory recursively remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            # Handle exceptions in case files can't be delete, such as if they are open
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                success = False

        print("Done emptying directory")

    # If the directory doesnt exist yet create it
    else:
        print(f"Creating '{directory_name}' directory...")
        os.makedirs(directory_name)
        print("Done creating directory")
    
    return success

def extract_frames(video_name = "Video.mp4"):
    """Function to extract frames from a video at a specified interval.

    Parameters:
        video_name: Name of the video to extract frames from. Default is Video.mp4

    Returns:
        List of extracted images
    """

    # Create videocapture object
    cap = cv2.VideoCapture(video_name)
    # Throw error if video was not found
    if not cap.isOpened():
        print(f"Error: Could not open {video_name}.")
        return []

    print(f"Processing '{video_name}'...")

    # Create list to store all extracted frames
    frame_list = []
    frame_count = 0
    
    # Get the total number of frames and fps for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize the progress bar with the total frame count
    with tqdm.tqdm(total=total_frames, desc="Extracting Frames", unit="frame") as pbar:
        frame_count = 0
        while True:
            # Extract the next frame from the video
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frames to the list
            frame_list.append(frame)
            
            frame_count += 1
            # Move the bar forward by 1 for every frame read
            pbar.update(1)

    cap.release()
    print(f"Done! Extracted {frame_count} frames.")
    return fps, frame_list

def create_video(frame_rate = 30, calibration_dir = "calibration_images", video_name = "constructed_video.mp4"):
    """This function takes a directory of images and converts them into a video.

    Parameters:
        frame_rate: The frame rate at which to create the video
        calibration_dir: Name of the directory from which to pull images for the video. 
                         Default is calibration_images.

    Returns:
        True if successful, otherwise False
    """

    print(f"Reading images from: {calibration_dir}")

    # Find all files matching pattern "frame_*.jpg" in the given directory
    # Use sorted() so frames are in the correct numerical order
    frame_files = sorted(glob.glob(os.path.join(calibration_dir, 'frame_*.jpg')))
    
    # If no frames were found, abort early
    if not frame_files:
        print(f"No image frames found in {calibration_dir}")
        return False
    
    # Read the first frame to determine image size
    first_frame = cv2.imread(frame_files[0])
    
    # If the first frame failed to load, abort
    if first_frame is None:
        print(f"Failed to read the first frame: {frame_files[0]}")
        # Return fail status
        return False
    
    # Get height and width from the image shape
    height, width = first_frame.shape[:2]
    
    # Construct full output path for the video file
    output_video = os.path.join(calibration_dir, video_name)
    
    # Define the codec used for encoding the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Initialize the VideoWriter object
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    # Loop over all frame file paths with a progress bar
    for frame_path in tqdm.tqdm(frame_files, desc="Creating video"):
        frame = cv2.imread(frame_path)
        
        # If the frame failed to load, skip it and continue
        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            continue
        # Write the frame into the video file
        out.write(frame)
    # Release the VideoWriter to finalize and properly save the file
    out.release()
    print(f"Video created and saved to: {output_video}")
    # Return success status
    return True

def plot_camera_movement(data, plot_dir, plot_name, legend_1, legend_2, legend_3):
    """
    Plots the X, Y, and Z translation of the camera over a sequence of frames.

    Parameters:
        data: List or array of camera positions, where each entry is [X, Y, Z]
        plot_dir: Directory to save the plot image
        plot_name: Filename for the saved plot
        legend_1: Label for X-axis data
        legend_2: Label for Y-axis data
        legend_3: Label for Z-axis data

    Returns:
        None (the function saves the plot as an image file)
    """

    # Convert list of arrays to a single (N, 3) numpy array
    positions = np.array(data)
    # Create a frame index array from 0 to N-1
    frames = np.arange(len(positions))

    plt.figure(figsize=(10, 5))
    # Plot X positions over frames in blue with the given label
    plt.plot(frames, positions[:, 0], label=legend_1, color='blue')
    # Plot Y positions over frames in green with the given label
    plt.plot(frames, positions[:, 1], label=legend_2, color='green')
    # Plot Z positions over frames in red with the given label
    plt.plot(frames, positions[:, 2], label=legend_3, color='red')

    plt.title("Camera Position per Calibration Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Distance (mm)")
    plt.legend()
    # Enable grid lines for easier reading
    plt.grid(True)
    
    # Define the save path
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Close the plot to prevent it from popping up and pausing the script
    plt.close()

def draw_axes_on_frame(image, mtx, dist, rvec, tvec, corners, checkerboard_dim, square_size):
    """
    Draws X, Y, and Z axes on a given image at the checkerboard origin.

    Parameters:
        image: The input image on which to draw the axes
        mtx: Camera intrinsic matrix (from calibration)
        dist: Camera distortion coefficients (from calibration)
        rvec: Rotation vector for the checkerboard pose
        tvec: Translation vector for the checkerboard pose
        corners: Detected 2D coordinates of the checkerboard corners
        checkerboard_dim: Dimensions of the checkerboard (rows, cols)
        square_size: Size of one checkerboard square (in same units as tvec)

    Returns:
        image: The input image with drawn XYZ axes.
    """
    
    # Define the 3D points for the axes (2 squares long)
    axis_len = square_size * 2
    axis_points = np.float32([[axis_len, 0, 0], 
                              [0, axis_len, 0], 
                              [0, 0, axis_len]])

    # Project 3D points to 2D image plane
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
    # Reshape the projected points to a simple (3, 2) array of integer pixel coordinates
    imgpts = imgpts.reshape(-1, 2).astype(int)

    # The origin of the axes is the first detected checkerboard corner
    origin = tuple(corners[0].ravel().astype(int))
    # Draw lines: X-Red, Y-Green, Z-Blue
    image = cv2.line(image, origin, tuple(imgpts[2]), (255, 0, 0), 3) # Z
    image = cv2.line(image, origin, tuple(imgpts[1]), (0, 255, 0), 3) # Y
    image = cv2.line(image, origin, tuple(imgpts[0]), (0, 0, 255), 3) # X
    
    # Return the image with drawn axes
    return image

def get_calibration(image_list, checkerboard_dim, square_size, frame_interval = 1, calibration_dir = "calibration_images"):
    """
    Detects checkerboard corners, performs camera calibration, saves a result video, 
    and returns the position history.

    Parameters:
        frame_interval: The interval at which frames are extracted. Default is 1 meaning every frame.

    """
    # Create directory to store calibration images
    if not create_directory(calibration_dir):
        return False, None, None

    print(f"Calibrating with extracted frames...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard_dim image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard_dim image
    imgpoints = []
    # Vector to store copies of images
    valid_frames = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard_dim[0]*checkerboard_dim[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2)
    objp *= square_size

    for image in tqdm.tqdm(image_list, desc="Detecting Corners"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
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
            valid_frames.append(image.copy())

    """
    Performing camera calibration by passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the detected corners (imgpoints)
    """
    if len(objpoints) == 0:
        print("No valid checkerboards detected.")
        return False, None, None

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints[::frame_interval], imgpoints[::frame_interval], gray.shape[::-1],None,None)
    print("Camera calibration complete!")

    world_coords = []
    rvec_history = []
    tvec_history = []

    if ret:
        # Draw everything and save
        for i in tqdm.tqdm(range(len(valid_frames)), desc="Processing Frames"):
            img = valid_frames[i]
            
            success, rvec, tvec = cv2.solvePnP(objp, imgpoints[i], mtx, dist)
            if not success:
                continue

            # print(f"\nResults for frame_{i}.jpg")
            # print("Rotation Vector (rvec):\n", rvec)
            # print("Translation Vector (tvec):\n", tvec)

            # Convert rotation vector to matrix for coordinate math
            rmat, _ = cv2.Rodrigues(rvec)

            # The position of the camera in world coordinates
            cam_pos = -np.matrix(rmat).T * np.matrix(tvec)
            # print("Camera Position in World Coords:\n", cam_pos)
            world_coords.append(cam_pos)
            rvec_history.append(rvec)
            tvec_history.append(tvec)

            # Draw the checkerboard pattern
            cv2.drawChessboardCorners(img, checkerboard_dim, imgpoints[i], True)
            
            # Draw axes using solvePnP pose
            img = draw_axes_on_frame(img, mtx, dist, rvec, tvec, imgpoints[i], checkerboard_dim, square_size)

            # Save the final combined image
            cv2.imwrite(os.path.join(calibration_dir, f"frame_{i:04d}.jpg"), img)

        # Save Camera Matrix (Intrinsics) to CSV
        np.savetxt("camera_matrix.txt", mtx, delimiter=",")
        # Save Distortion Coefficients to CSV
        np.savetxt("distortion_coefficients.txt", dist, delimiter=",")
        
        print("Intrinsics saved to 'camera_matrix.txt' and 'distortion_coefficients.txt'")

        plot_dir="plots"
        create_directory(plot_dir)
        plot_camera_movement(tvec_history, plot_dir=plot_dir, plot_name="camera_translation_plot.jpg", legend_1 = "X", legend_2 = "Y", legend_3 = "Z")

        return True, mtx, dist
    else:
        print("Calibration failed.")
        return False, None, None
    
def track_object_pose_sift(video_name, reference_image, real_width_mm, real_height_mm, output_dir="pose_output", calib_dir="."):

    # Tracks a planar object using SIFT + solvePnP.

    # Parameters:
    #     video_name: input video
    #     reference_image: image of known planar object
    #     real_width_mm: real width of object
    #     real_height_mm: real height of object


    # Load calibration and distortion
    mtx = np.loadtxt(os.path.join(calib_dir, "camera_matrix.txt"), delimiter=",")
    dist = np.loadtxt(os.path.join(calib_dir, "distortion_coefficients.txt"), delimiter=",")
    dist = dist.reshape(-1, 1) if dist.ndim == 1 else dist

    # Load reference img that wil be used for SIFT 
    ref_img = cv2.imread(reference_image)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    #creat sift detector
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)

    # set up feature matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 3D object center
    objp = np.array([
        [-real_width_mm/2, -real_height_mm/2, 0],
        [ real_width_mm/2, -real_height_mm/2, 0],
        [ real_width_mm/2,  real_height_mm/2, 0],
        [-real_width_mm/2,  real_height_mm/2, 0]
    ], dtype=np.float32)

    # extract frames 
    fps, frames = extract_frames(video_name)
    create_directory(output_dir)

    pose_history = []
    frame_count = 0

    print("Running SIFT + solvePnP tracking...")
    # for loop for each frame in the video
    for i, img in enumerate(tqdm.tqdm(frames)):

        frame_count += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp_frame, des_frame = sift.detectAndCompute(gray, None)
        if des_frame is None:
            continue

        #  Match and filter
        matches = flann.knnMatch(des_ref, des_frame, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            continue

        #  Get points 
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        #  Homography 
        H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue

        inliers = np.sum(mask)
        if inliers < 10:
            continue
        #  Project corners 
        h_ref, w_ref = gray_ref.shape
        corners = np.float32([
            [0,0],
            [w_ref,0],
            [w_ref,h_ref],
            [0,h_ref]
        ]).reshape(-1,1,2)

        img_pts = cv2.perspectiveTransform(corners, H)

        #  Solve for position using solvePnP 
        success, rvec, tvec = cv2.solvePnP(objp, img_pts, mtx, dist)
        if not success:
            continue

        X, Y, Z = tvec.flatten()
        pose_history.append([X, Y, Z])

        #  Draw box 
        img_pts_int = np.int32(img_pts)
        cv2.polylines(img, [img_pts_int], True, (0,255,0), 3)
    
    

        success, rvec, tvec = cv2.solvePnP(objp, img_pts, mtx, dist)
        if np.any(np.isnan(tvec)) or np.any(np.isinf(tvec)):
            continue

        #  Text 
        text = f"X:{X:.1f} Y:{Y:.1f} Z:{Z:.1f} mm"
        cv2.putText(img, text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        #  Save frame 
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), img)

    #  Plot 
    plot_dir = "plots_pose"
    create_directory(plot_dir)

    if len(pose_history) > 0:
        plot_camera_movement(pose_history, plot_dir=plot_dir, plot_name="pose_plot.jpg", legend_1="X", legend_2="Y", legend_3="Z")

    print("Tracking complete.")
    return fps



if __name__ == "__main__":
    frame_interval = 40

    # Extract calibration images from video
    fps, video_frames = extract_frames(video_name = "Video.mp4")

    # Get camera intrinsics (matrix and distortion) and solve for world coordinates
    get_calibration(frame_interval = frame_interval, image_list = video_frames, checkerboard_dim = (10,7), square_size = 25, calibration_dir = "calibration_images")

    # Make a video from the images in a directory
    create_video(frame_rate = fps, calibration_dir = "calibration_images", video_name = "constructed_video.mp4")

    # runs the object position tracking code
    fps = track_object_pose_sift( video_name="book.mp4", reference_image="reference.jpg", real_width_mm=127, real_height_mm=191, output_dir="pose_output", calib_dir=".")
# creats the video from the saved frames
    create_video(frame_rate=fps, calibration_dir="pose_output", video_name="pose_output.mp4")