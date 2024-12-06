import numpy as np
from scipy import signal
from scipy.ndimage import shift as nd_shift

def apply_flip(mask, flip_axes):
    """
    Flip the mask along specified axes.

    Parameters:
    - mask (np.ndarray): 3D binary mask.
    - flip_axes (tuple): Axes to flip (0 for X, 1 for Y, 2 for Z).

    Returns:
    - np.ndarray: Flipped mask.
    """
    return np.flip(mask, axis=flip_axes)

def find_best_shift(mask1, mask2_flipped):
    """
    Find the shift that maximizes the overlap between mask1 and mask2_flipped.

    Parameters:
    - mask1 (np.ndarray): Reference 3D binary mask.
    - mask2_flipped (np.ndarray): Flipped 3D binary mask to align.

    Returns:
    - tuple: Best shift along (X, Y, Z).
    - float: Maximum overlap value.
    """
    # Compute cross-correlation
    cross_corr = signal.correlate(mask1, mask2_flipped, mode='full')
    
    # Find the index of the maximum correlation
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    
    # Calculate the shift relative to the center
    shifts = np.array(max_idx) - np.array(mask1.shape) + 1
    
    # Calculate the maximum overlap
    max_overlap = cross_corr[max_idx]
    
    return tuple(shifts), max_overlap

def shift_mask(mask, shift):
    """
    Shift the mask by the specified amount with zero-padding.

    Parameters:
    - mask (np.ndarray): 3D binary mask.
    - shift (tuple): Shift along (X, Y, Z). Positive values shift towards higher indices.

    Returns:
    - np.ndarray: Shifted mask.
    """
    shifted_mask = np.zeros_like(mask)
    
    # Calculate slicing indices for source and destination
    src_x_start = max(0, -shift[0])
    src_x_end = mask.shape[0] - max(0, shift[0])
    dst_x_start = max(0, shift[0])
    dst_x_end = mask.shape[0] - max(0, -shift[0])
    
    src_y_start = max(0, -shift[1])
    src_y_end = mask.shape[1] - max(0, shift[1])
    dst_y_start = max(0, shift[1])
    dst_y_end = mask.shape[1] - max(0, -shift[1])
    
    src_z_start = max(0, -shift[2])
    src_z_end = mask.shape[2] - max(0, shift[2])
    dst_z_start = max(0, shift[2])
    dst_z_end = mask.shape[2] - max(0, -shift[2])
    
    # Apply the shift
    shifted_mask[dst_x_start:dst_x_end, dst_y_start:dst_y_end, dst_z_start:dst_z_end] = \
        mask[src_x_start:src_x_end, src_y_start:src_y_end, src_z_start:src_z_end]
    
    return shifted_mask

def align_masks(mask1, mask2):
    """
    Align mask2 to mask1 by flipping along axes and shifting to maximize overlap.

    Parameters:
    - mask1 (np.ndarray): Reference 3D binary mask.
    - mask2 (np.ndarray): 3D binary mask to be aligned.

    Returns:
    - np.ndarray: Aligned mask2.
    - dict: Alignment details including flip axes and shifts.
    """
    # Define all possible flip combinations (no flip, flip X, flip Y, flip Z, etc.)
    flip_combinations = [
        (),                 # No flip
        (0,),               # Flip X
        (1,),               # Flip Y
        (2,),               # Flip Z
        (0, 1),             # Flip X and Y
        (0, 2),             # Flip X and Z
        (1, 2),             # Flip Y and Z
        (0, 1, 2)           # Flip X, Y, and Z
    ]
    
    best_overlap = -1
    best_shift = (0, 0, 0)
    best_flip = ()
    best_aligned_mask = None
    
    for flip_axes in flip_combinations:
        # Apply flipping
        if flip_axes:
            mask2_flipped = apply_flip(mask2, flip_axes)
        else:
            mask2_flipped = mask2.copy()
        
        # Find the best shift for this flipped mask
        shift, overlap = find_best_shift(mask1, mask2_flipped)
        
        # Update best parameters if this is the best overlap so far
        if overlap > best_overlap:
            best_overlap = overlap
            best_shift = shift
            best_flip = flip_axes
            # Shift the flipped mask to get the aligned mask
            best_aligned_mask = shift_mask(mask2_flipped, shift)
    
    alignment_details = {
        'flip_axes': best_flip,
        'shift': best_shift,
        'max_overlap': best_overlap
    }
    
    return best_aligned_mask, alignment_details

# Example Usage
if __name__ == "__main__":
    # Example masks (replace with actual data)
    # mask1 and mask2 should be 3D numpy arrays with values 0 or 1
    # For demonstration, create simple masks
    mask1 = np.zeros((100, 100, 100), dtype=np.uint8)
    mask1[30:70, 30:70, 30:70] = 1  # A cube in the center

    mask2 = np.zeros((100, 100, 100), dtype=np.uint8)
    mask2[40:80, 40:80, 40:80] = 1  # A shifted cube

    # Align mask2 to mask1
    aligned_mask2, details = align_masks(mask1, mask2)
    
    print("Alignment Details:")
    print(f" - Flip Axes: {details['flip_axes']}")
    print(f" - Shift: {details['shift']}")
    print(f" - Max Overlap: {details['max_overlap']}")
    
    # Verify alignment by computing overlap
    overlap = np.sum(mask1 * aligned_mask2)
    print(f"Total Overlapping Voxels: {overlap}")
