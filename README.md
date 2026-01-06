# Polaroid Image Processor

An automatic processor for high-resolution Polaroid scans that detects, crops, and corrects perspective distortion to extract perfectly aligned Polaroid images.

## Features

- **Automatic Detection**: Finds Polaroid boundaries using edge detection and contour analysis
- **Perspective Correction**: Corrects for scanning angle and slight warping
- **Barrel Distortion Correction**: Fixes lens-like distortion from non-flat Polaroids in holders
- **Edge Shadow Compensation**: Removes holder-induced shadows at image borders
- **Safety Margin Cropping**: Prevents black edge artifacts with configurable inward cropping
- **Exact Aspect Ratio**: Outputs images in the correct Polaroid aspect ratio (88x107mm)
- **High Resolution**: Preserves original 600 DPI resolution
- **Batch Processing**: Process entire directories of TIFF files
- **Quality Control**: Validates detection quality and provides debug visualizations

## Requirements

- Python 3.8+
- OpenCV
- Pillow (PIL)
- NumPy
- PyYAML

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Single Image Processing

```bash
# Process single image (creates both TIFF and JPG)
python main.py input_image.tif --single

# Create only JPG output
python main.py input_image.tif --single --jpg-only

# Create only TIFF output
python main.py input_image.tif --single --no-jpg
```

### Batch Processing

```bash
# Process all images (creates both TIFF and JPG in separate directories)
python main.py input_directory/

# Create only JPG files
python main.py input_directory/ --jpg-only

# Create only TIFF files
python main.py input_directory/ --no-jpg

# Custom output directory
python main.py input_directory/ -o my_output/
```

### Distortion and Shadow Correction Options

```bash
# Disable distortion correction
python main.py input_directory/ --no-distortion-correction

# Override distortion coefficient (negative for barrel, positive for pincushion)
python main.py input_directory/ --distortion-k1 -0.2

# Override safety margin for cropping (percentage)
python main.py input_directory/ --safety-margin 0.8

# Disable shadow correction
python main.py input_directory/ --no-shadow-correction

# Override shadow correction brightness boost
python main.py input_directory/ --shadow-boost 50
```

### Output Directory Structure

- **Default mode**: Creates both `output_fullres/` (TIFF files) and `output_jpg/` (JPG files)
- **JPG-only mode**: Creates only `output_jpg/` directory
- **TIFF-only mode**: Creates only `output_fullres/` directory

### Configuration

Edit `config/processing_params.yaml` to adjust detection parameters for your specific scan setup.

## How It Works

1. **Preprocessing**: Converts image to grayscale and applies thresholding to separate white Polaroid borders from black background
2. **Edge Detection**: Uses Canny edge detection to find boundaries
3. **Contour Analysis**: Finds rectangular contours and filters by area and aspect ratio
4. **Corner Detection**: Identifies the four corners of the Polaroid
5. **Perspective Correction**: Applies perspective transformation with safety margin cropping
6. **Distortion Correction**: Applies radial undistortion to fix barrel/pincushion distortion from holders
7. **Shadow Correction**: Brightens shadowed edges and corners caused by holder shadows
8. **Output**: Saves the corrected image with exact Polaroid dimensions (88x107mm at 600 DPI)

## Output

- **TIFF Images**: High-resolution corrected Polaroid images (preserves original resolution)
- **JPG Images**: 350 DPI JPG files at 2x original Polaroid size
- **Debug Images**: Visualization showing detected corners (if enabled in config)
- **Quality Scores**: Confidence metrics for each detection

### File Specifications

- **TIFF Output**: High resolution (typically 20,000+ pixels), perfect aspect ratio (88:107mm)
- **JPG Output**: 350 DPI, 2x original Polaroid size, 95% quality

## Configuration Options

Key parameters you can adjust in `config/processing_params.yaml`:

### Detection Parameters
- `edge_detection.canny_low/high`: Edge detection sensitivity
- `contour_detection.min_area`: Minimum size for detected rectangles
- `quality_control.min_quality_score`: Minimum confidence threshold
- `image.dpi`: Input image DPI (default: 600)

### Distortion Correction (for holder-induced barrel distortion)
- `distortion_correction.enabled`: Enable/disable distortion correction
- `distortion_correction.k1`: Primary radial distortion coefficient (-0.3 to 0.3)
  - Negative values correct barrel distortion (bulge outward)
  - Positive values correct pincushion distortion (pinch inward)
  - Start with -0.15 for typical holder distortion

### Cropping Safety Margin
- `cropping.safety_margin_percent`: Crop inward from detected edge (0.3-1.5%)
  - Prevents black edge artifacts from imperfect detection

### Shadow Correction (for holder-induced edge shadows)
- `shadow_correction.enabled`: Enable/disable shadow correction
- `shadow_correction.border_region_percent`: Width of border region to analyze
- `shadow_correction.max_brightness_boost`: Maximum brightness increase (0-100)
- `shadow_correction.detection_threshold`: Brightness ratio trigger (default: 0.85)

## Troubleshooting

- **No Polaroid Detected**: Try adjusting Canny thresholds or minimum area in config
- **Poor Quality Detection**: Enable debug visualization to see what's being detected
- **Wrong Aspect Ratio**: Verify your input images have the expected black background/white border contrast
- **Barrel Distortion (bulging)**: Increase negative k1 value (e.g., -0.2 to -0.3)
- **Pincushion Distortion**: Use positive k1 values (e.g., 0.1 to 0.2)
- **Black Edge Artifacts**: Increase `safety_margin_percent` (e.g., 0.8 to 1.5)
- **Dark Corners/Edges**: Increase `max_brightness_boost` or lower `detection_threshold`
- **Over-brightened Borders**: Decrease `max_brightness_boost` or increase `detection_threshold`

### Tuning Distortion Correction

1. Start with `k1 = -0.15` (default)
2. Process a test image with straight lines (buildings, horizons)
3. If lines still curve outward: increase magnitude (e.g., -0.2, -0.25)
4. If lines curve inward (overcorrected): decrease magnitude (e.g., -0.1, -0.05)
5. Once tuned, the same values work for all images from the same holder

## File Structure

```
rollerroids/
├── src/
│   ├── polaroid_processor.py    # Main processing class
│   ├── edge_detector.py         # Edge detection utilities
│   ├── image_utils.py           # Image utilities & distortion correction
│   └── shadow_corrector.py      # Edge shadow compensation
├── config/
│   └── processing_params.yaml   # Configuration file
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Example

```bash
# Process a single image
python main.py scan_001.tif --single

# Process all TIFF files in a directory
python main.py scans/ -o processed/

# Process with custom distortion and shadow settings
python main.py scans/ --distortion-k1 -0.2 --shadow-boost 40

# Use custom configuration
python main.py scans/ -o processed/ -c my_config.yaml
```
