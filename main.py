#!/usr/bin/env python3
"""
Main script for Polaroid image processing.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import cv2
from PIL import Image

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from polaroid_processor import PolaroidProcessor  # noqa: E402
from edge_detector import EdgeDetector  # noqa: E402
from image_utils import (  # noqa: E402
    create_debug_visualization,
    calculate_quality_score,
    apply_barrel_distortion_correction,
    get_distortion_correction_config,
    apply_safety_margin_crop
)
from shadow_corrector import ShadowCorrector, get_shadow_correction_config  # NOQA


def load_config(config_path: str = "config/processing_params.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error loading config: {e}")
        return {}


def process_single_image(
    input_path: str,
    output_path: str,
    config: dict,
    create_jpg: bool = True,
    jpg_only: bool = False
) -> bool:
    """Process a single Polaroid scan image."""
    # Initialize processor with full config
    processor = PolaroidProcessor(config=config)

    # Initialize edge detector with config
    edge_config = config.get('edge_detection', {})
    edge_detector = EdgeDetector(
        canny_low=edge_config.get('canny_low', 50),
        canny_high=edge_config.get('canny_high', 150)
    )

    try:
        # Load and preprocess image
        image = processor.load_image(input_path)
        processed = processor.preprocess_image(image)

        # Detect edges and find contours
        edges = edge_detector.detect_edges(processed)
        contours = edge_detector.find_rectangular_contours(
            edges,
            min_area=processor.min_contour_area
        )

        # Find best matching rectangle using aspect ratio from config
        corners = edge_detector.find_largest_rectangle(
            contours,
            target_aspect_ratio=processor.aspect_ratio
        )

        if corners is None:
            print(f"Error: Could not detect Polaroid in {input_path}")
            return False

        # Calculate quality score using aspect ratio from config
        quality_score = calculate_quality_score(
            image, corners, target_aspect_ratio=processor.aspect_ratio
        )
        min_quality = config.get(
            'quality_control', {}
        ).get('min_quality_score', 0.3)

        if quality_score < min_quality:
            print(
                f"Warning: Low quality detection (score: {quality_score:.2f}) "
                f"for {input_path}"
            )

        # Apply perspective correction (without safety margin - applied later)
        corrected = processor.perspective_correct(image, corners)

        # Apply barrel distortion correction if enabled
        # (BEFORE safety margin crop)
        dist_cfg = get_distortion_correction_config(config)
        distortion_enabled, k1, k2, k3 = dist_cfg
        if distortion_enabled:
            corrected = apply_barrel_distortion_correction(
                corrected, k1, k2, k3
            )
            print(f"Distortion correction applied: k1={k1}, k2={k2}")

        # Apply safety margin crop AFTER distortion correction
        # This removes any black edges introduced by distortion correction
        cropping_config = config.get('cropping', {})
        safety_margin = cropping_config.get('safety_margin_percent', 0.0)
        if safety_margin > 0:
            corrected = apply_safety_margin_crop(corrected, safety_margin)
            print(f"Safety margin crop applied: {safety_margin}%")

        # Apply shadow correction if enabled
        shadow_enabled, shadow_params = get_shadow_correction_config(config)
        if shadow_enabled:
            shadow_corrector = ShadowCorrector(**shadow_params)
            corrected = shadow_corrector.apply_correction(corrected)
            print("Applied edge shadow correction")

        # Save TIFF result (unless JPG-only mode)
        if not jpg_only:
            from image_utils import save_with_metadata
            save_with_metadata(corrected, output_path, processor.dpi)
            print(f"Successfully processed TIFF: {output_path}")

        # Create JPG output if requested
        if create_jpg or jpg_only:
            jpg_output = processor.create_jpg_output(corrected)
            jpg_rgb = cv2.cvtColor(jpg_output, cv2.COLOR_BGR2RGB)
            jpg_pil = Image.fromarray(jpg_rgb)

            # Use output_path directly if JPG-only,
            # otherwise create JPG filename
            if jpg_only:
                jpg_path = output_path
            else:
                jpg_path = output_path.replace(
                    '.tif', '.jpg'
                ).replace('.tiff', '.jpg')

            jpg_pil.save(
                jpg_path,
                "JPEG",
                quality=processor.jpg_quality,
                dpi=(processor.jpg_dpi, processor.jpg_dpi)
            )
            print(f"Successfully processed JPG: {jpg_path}")

        # Create debug visualization if enabled
        qc_config = config.get('quality_control', {})
        if qc_config.get('enable_debug_visualization', False):
            debug_dir = qc_config.get('debug_output_dir', 'debug')
            os.makedirs(debug_dir, exist_ok=True)

            debug_filename = f"debug_{os.path.basename(input_path)}"
            debug_path = os.path.join(debug_dir, debug_filename)
            create_debug_visualization(image, corners, debug_path)

        print(
            f"Successfully processed: {input_path} -> {output_path} "
            f"(quality: {quality_score:.2f})"
        )
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_batch(
    input_dir: str,
    output_dir: str,
    config: dict,
    create_jpg: bool = True,
    jpg_only: bool = False
) -> None:
    """Process all TIFF files in a directory."""
    print('-' * 75)
    print('Starting batch processing...')
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # If creating both TIFF and JPG, create separate directories
    if create_jpg and not jpg_only:
        jpg_dir = output_dir.replace('output_fullres', 'output_jpg')
        os.makedirs(jpg_dir, exist_ok=True)
        print(f"TIFF output: {output_dir}")
        print(f"JPG output: {jpg_dir}")
    else:
        print(f"Output: {output_dir}")

    # Find TIFF files (avoid duplicates)
    tiff_files = set()
    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        for file_path in Path(input_dir).glob(f"*{ext}"):
            tiff_files.add(file_path)

    tiff_files = sorted(list(tiff_files))

    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return

    print(f"Found {len(tiff_files)} TIFF files to process")

    successful = 0
    failed = 0
    print('-' * 75)
    for tiff_file in tiff_files:
        print(f'Processing {tiff_file.name}...')
        input_path = str(tiff_file)
        output_filename = f"processed_{tiff_file.name}"
        output_path = os.path.join(output_dir, output_filename)

        # For batch processing with both TIFF and JPG
        if create_jpg and not jpg_only:
            # Process with TIFF output
            if process_single_image(
                input_path, output_path, config,
                create_jpg=False, jpg_only=False
            ):
                # Now create JPG in separate directory
                jpg_dir = output_dir.replace('output_fullres', 'output_jpg')
                jpg_filename = f"processed_{tiff_file.stem}.jpg"
                jpg_path = os.path.join(jpg_dir, jpg_filename)

                # Load the processed TIFF and create JPG
                processor = PolaroidProcessor(config=config)
                processed_tiff = processor.load_image(output_path)
                jpg_output = processor.create_jpg_output(processed_tiff)
                jpg_rgb = cv2.cvtColor(jpg_output, cv2.COLOR_BGR2RGB)
                jpg_pil = Image.fromarray(jpg_rgb)
                jpg_pil.save(
                    jpg_path,
                    "JPEG",
                    quality=processor.jpg_quality,
                    dpi=(processor.jpg_dpi, processor.jpg_dpi)
                )
                print(f"Successfully processed JPG: {jpg_path}")

                successful += 1
            else:
                failed += 1
        else:
            # Single output mode (TIFF-only or JPG-only)
            if jpg_only:
                jpg_output_path = output_path.replace(
                    '.tif', '.jpg'
                ).replace('.tiff', '.jpg')
                if process_single_image(
                    input_path, jpg_output_path, config,
                    create_jpg=True, jpg_only=True
                ):
                    successful += 1
                else:
                    failed += 1
            else:
                if process_single_image(
                    input_path, output_path, config,
                    create_jpg=create_jpg, jpg_only=jpg_only
                ):
                    successful += 1
                else:
                    failed += 1
        print('-' * 75)

    print("\nBatch processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Polaroid scans with automatic detection, "
                    "distortion correction, and shadow compensation"
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "-c", "--config",
        default="config/processing_params.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Process single file (default: batch mode)"
    )
    parser.add_argument(
        "--jpg-only", action="store_true",
        help="Create only JPG output (no TIFF)"
    )
    parser.add_argument(
        "--no-jpg", action="store_true",
        help="Create only TIFF output (no JPG)"
    )

    # Distortion correction options
    parser.add_argument(
        "--no-distortion-correction", action="store_true",
        help="Disable barrel distortion correction"
    )
    parser.add_argument(
        "--distortion-k1", type=float, default=None,
        help="Override k1 distortion coefficient "
             "(negative=barrel, positive=pincushion)"
    )

    # Cropping options
    parser.add_argument(
        "--safety-margin", type=float, default=None,
        help="Override cropping safety margin (percentage, e.g., 0.5)"
    )

    # Shadow correction options
    parser.add_argument(
        "--no-shadow-correction", action="store_true",
        help="Disable edge shadow correction"
    )
    parser.add_argument(
        "--shadow-boost", type=int, default=None,
        help="Override max brightness boost for shadow correction (0-100)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Handle distortion correction overrides
    if args.no_distortion_correction:
        if 'distortion_correction' not in config:
            config['distortion_correction'] = {}
        config['distortion_correction']['enabled'] = False
        print("Distortion correction disabled via command line")
    elif args.distortion_k1 is not None:
        if 'distortion_correction' not in config:
            config['distortion_correction'] = {}
        config['distortion_correction']['enabled'] = True
        config['distortion_correction']['k1'] = args.distortion_k1
        print(f"Distortion k1 set to {args.distortion_k1} via command line")

    # Handle safety margin override
    if args.safety_margin is not None:
        if 'cropping' not in config:
            config['cropping'] = {}
        config['cropping']['safety_margin_percent'] = args.safety_margin
        print(f"Safety margin set to {args.safety_margin}% via command line")

    # Handle shadow correction overrides
    if args.no_shadow_correction:
        if 'shadow_correction' not in config:
            config['shadow_correction'] = {}
        config['shadow_correction']['enabled'] = False
        print("Shadow correction disabled via command line")
    elif args.shadow_boost is not None:
        if 'shadow_correction' not in config:
            config['shadow_correction'] = {}
        config['shadow_correction']['enabled'] = True
        config['shadow_correction']['max_brightness_boost'] = args.shadow_boost
        print(f"Shadow boost set to {args.shadow_boost} via command line")

    # Determine JPG creation setting
    create_jpg = True
    if args.jpg_only:
        create_jpg = True
        print("JPG-only mode: Creating only JPG output")
    elif args.no_jpg:
        create_jpg = False
        print("TIFF-only mode: Creating only TIFF output")

    # Determine output paths
    if args.single:
        input_path = Path(args.input)
        if args.jpg_only:
            output_path = str(
                input_path.parent / f"processed_{input_path.stem}.jpg"
            )
        else:
            output_path = str(
                input_path.parent / f"processed_{input_path.name}"
            )
    else:
        if args.output:
            if args.jpg_only:
                output_path = os.path.join(args.output, "output_jpg")
            else:
                output_path = os.path.join(args.output, "output_fullres")
        else:
            if args.jpg_only:
                output_path = "output_jpg"
            else:
                output_path = "output_fullres"

    # Process based on mode
    if args.single:
        if not os.path.isfile(args.input):
            print(f"Error: Input file {args.input} does not exist")
            sys.exit(1)

        success = process_single_image(
            args.input, output_path, config, create_jpg, args.jpg_only
        )
        sys.exit(0 if success else 1)
    else:
        if not os.path.isdir(args.input):
            print(f"Error: Input directory {args.input} does not exist")
            sys.exit(1)

        process_batch(
            args.input, output_path, config, create_jpg, args.jpg_only
        )


if __name__ == "__main__":
    main()
