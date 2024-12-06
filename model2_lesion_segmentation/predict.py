from pathlib import Path
import torch
import monai
from monai.transforms import ScaleIntensity, Spacing, ResizeWithPadOrCrop, ToTensor, EnsureType
import nibabel as nib
import numpy as np
import os
import warnings
from model import DualNet_seperate_load
from data_loader import general_transform
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def main():
    # Silence warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Choose dataset type: 'Example' or 'Original'
    dataset = 'Original'

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Basic parameters
    crop_size = [128, 128, 64]
    resample_spacing = [0.6640625, 0.6640625, 1.5]

    # Setup paths
    current_path = Path(__file__).parent
    root_path = current_path.parent
    data_path = root_path / 'data'

    # For the 'Original' dataset, we assume data is in data/Test/MRIs and data/Test/Prostates
    if dataset == 'Example':
        # Example dataset paths (if you have them)
        # Adjust if you have a different structure for 'Example'
        test_image_dir = data_path / 'Test' / 'Cropped_MRIs'
        test_label_dir = data_path / 'Test' / 'Cropped_Legions'
    elif dataset == 'Original':
        test_image_dir = data_path / 'Test' / 'Cropped_MRIs'
        test_label_dir = data_path / 'Test' / 'Cropped_Legions'
    else:
        raise ValueError("dataset must be 'Example' or 'Original'.")

    # Get test images and labels
    test_images = sorted(list(test_image_dir.glob('*.nii.gz')))
    test_labels = sorted(list(test_label_dir.glob('*.nii.gz')))

    if len(test_images) != len(test_labels):
        raise ValueError("Number of test images and labels must be the same.")

    test_dicts = [{'image': str(img), 'label': str(lbl)} for img, lbl in zip(test_images, test_labels)]

    # Define custom transforms
    transform = general_transform(crop_size, resample_spacing)

    test_dataset = monai.data.Dataset(data=test_dicts, transform=transform)
    test_loader = monai.data.DataLoader(test_dataset, batch_size=1)
    print('Loader created')

    # Load model
    # Make sure the UNET_Transformer_model_prostate.pt file is in the current directory
    model_path = current_path / 'UNET_Transformer_model_legion.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = DualNet_seperate_load(device=device, crop_patch_size=crop_size, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print('Model loaded from', model_path)

    # Create base directory for predicted labels
    pred_dir = current_path / 'predicted_labels'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organized outputs
    output_dir = pred_dir / 'output'
    segmentation_dir = pred_dir / 'segmentation'
    truth_dir = pred_dir / 'truth'
    image_dir = pred_dir / 'image'

    output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, percentile=95)

    # Lists to store per-sample metrics
    dice_scores = []
    hausdorff_distances = []

    # Prediction loop
    for i, batch in enumerate(test_loader):
        # Extract the original image volume information
        original_image = nib.load(test_dicts[i]['image'])
        original_affine = original_image.affine
        original_size = original_image.shape

        # Determine flip dimensions if needed (currently commented out)
        flip_dimensions = []
        # if you need orientation correction, adjust here:
        # for dim in range(3):
        #     if original_affine[dim][dim] < 0:
        #         flip_dimensions.append(dim + 2)

        image = batch['image'].to(device)
        if len(flip_dimensions) > 0:
            image = torch.flip(image, dims=flip_dimensions)

        # Perform prediction
        with torch.no_grad():
            output = model(image)
            output = torch.sigmoid(output)
        
        # Post-processing: Threshold to get binary segmentation
        output_binary = (output > 0.5).float()

        # Ensure ground truth is binary
        truth = batch['label'].float().to(device)
        truth_binary = (truth > 0.5).float()

        # Compute Dice Score
        dice = dice_metric(output_binary, truth_binary)
        dice_score = dice.item()
        dice_scores.append(dice_score)

        # Compute Hausdorff Distance
        hausdorff = hausdorff_metric(output_binary, truth_binary)
        hausdorff_distance = hausdorff.item()
        hausdorff_distances.append(hausdorff_distance)

        # Move tensors to CPU and convert to NumPy for saving
        output_np = output_binary.squeeze().cpu().numpy()
        truth_np = truth_binary.squeeze().cpu().numpy()
        image_np = image.squeeze().cpu().numpy()

        # Construct output filename
        image_filename = os.path.basename(test_dicts[i]['image'])
        if dataset == 'Example':
            # Example naming strategy if needed
            # Adjust if you have a specific naming format
            identifier = image_filename.replace('.nii.gz', '')
            output_filename = f'{identifier}_prediction.nii.gz'
        elif dataset == 'Original':
            # Assuming filenames like 'imageXXXX.nii.gz'
            if 'image' in image_filename:
                identifier = image_filename.split('image')[1].split('.nii.gz')[0]
                output_filename = f'image{identifier}_prediction.nii.gz'
            else:
                identifier = image_filename.replace('.nii.gz', '')
                output_filename = f'{identifier}_prediction.nii.gz'

        # Save outputs
        nifti_output = nib.Nifti1Image(output_np, affine=original_affine)
        nifti_segmentation = nib.Nifti1Image(output_np.astype(np.uint8), affine=original_affine)
        nifti_truth = nib.Nifti1Image(truth_np.astype(np.uint8), affine=original_affine)
        nifti_image = nib.Nifti1Image(image_np, affine=original_affine)

        # Save files in respective directories
        nib.save(nifti_output, str(output_dir / output_filename))
        nib.save(nifti_segmentation, str(segmentation_dir / f'segmentation_{output_filename}'))
        nib.save(nifti_truth, str(truth_dir / f'truth_{output_filename}'))
        nib.save(nifti_image, str(image_dir / f'image_{output_filename}'))

        print(f'Saved {output_filename} to organized directories')

    # Aggregate and print metrics
    dice_mean = np.mean(dice_scores) if dice_scores else float('nan')
    dice_std = np.std(dice_scores) if dice_scores else float('nan')
    hausdorff_mean = np.mean(hausdorff_distances) if hausdorff_distances else float('nan')
    hausdorff_std = np.std(hausdorff_distances) if hausdorff_distances else float('nan')

    print("\n=== Evaluation Metrics ===")
    print(f"Dice Score: Mean = {dice_mean:.4f}, Std = {dice_std:.4f}")
    print(f"Hausdorff Distance (95th percentile): Mean = {hausdorff_mean:.4f}, Std = {hausdorff_std:.4f}")

    # Optionally, save metrics to a text file
    metrics_file = pred_dir / 'evaluation_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("=== Evaluation Metrics ===\n")
        f.write(f"Dice Score: Mean = {dice_mean:.4f}, Std = {dice_std:.4f}\n")
        f.write(f"Hausdorff Distance (95th percentile): Mean = {hausdorff_mean:.4f}, Std = {hausdorff_std:.4f}\n")
    
    print(f"Saved evaluation metrics to {metrics_file}")

if __name__ == '__main__':
    main()
