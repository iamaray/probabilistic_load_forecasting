import os
import json
import re
import csv
from collections import defaultdict
import numpy as np


def analyze_results(results_dir="results"):
    """
    Analyze the results directory to find the best models (ACR closest to 0.8 and lowest AES)
    for each distribution type and spatial/non-spatial category.
    """
    # Dictionary to store results by category and distribution type
    models = defaultdict(lambda: defaultdict(list))

    # Target ACR value
    target_acr = 0.8

    # Regular expression to extract information from filenames
    pattern = r"metrics_(spatial|non_spatial)_([a-z_]+)_n_(\d+)_device_([a-z]+)(?:_nu_([0-9.]+))?\.json"

    # Iterate through all files in the results directory
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(results_dir, filename)
        match = re.match(pattern, filename)

        if not match:
            print(f"Skipping file with unrecognized format: {filename}")
            continue

        # Extract information from filename
        spatial_type, dist_type, n_components, device, nu = match.groups()

        # Read the metrics file
        try:
            with open(filepath, 'r') as f:
                metrics = json.load(f)

            # Extract ACR and calculate distance to target
            acr = metrics.get('ACR')
            aes = metrics.get('AES')
            if acr is None or aes is None:
                print(f"Missing ACR or AES in {filename}")
                continue

            acr_distance = abs(acr - target_acr)

            # Store model information
            model_info = {
                'filename': filename,
                'n_components': int(n_components),
                'device': device,
                'nu': float(nu) if nu else None,
                'metrics': metrics,
                'acr_distance': acr_distance
            }

            # Group by spatial type and distribution type
            models[spatial_type][dist_type].append(model_info)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Find the best model for each category and distribution type
    best_models_acr = {}
    best_models_aes = {}
    all_acr_values = []

    for spatial_type, dist_types in models.items():
        best_models_acr[spatial_type] = {}
        best_models_aes[spatial_type] = {}

        for dist_type, model_list in dist_types.items():
            # Sort by ACR distance (closest to target)
            sorted_by_acr = sorted(model_list, key=lambda x: x['acr_distance'])
            # Sort by AES (lowest)
            sorted_by_aes = sorted(
                model_list, key=lambda x: x['metrics']['AES'])

            # Collect all ACR values for statistics
            for model in model_list:
                all_acr_values.append(model['metrics']['ACR'])

            if sorted_by_acr:
                best_model_acr = sorted_by_acr[0]
                best_models_acr[spatial_type][dist_type] = {
                    'filename': best_model_acr['filename'],
                    'n_components': best_model_acr['n_components'],
                    'nu': best_model_acr['nu'],
                    'metrics': best_model_acr['metrics'],
                    'acr_distance': best_model_acr['acr_distance']
                }

            if sorted_by_aes:
                best_model_aes = sorted_by_aes[0]
                best_models_aes[spatial_type][dist_type] = {
                    'filename': best_model_aes['filename'],
                    'n_components': best_model_aes['n_components'],
                    'nu': best_model_aes['nu'],
                    'metrics': best_model_aes['metrics'],
                    'acr_distance': best_model_aes['acr_distance']
                }

    # Calculate ACR statistics
    acr_stats = {
        'min': min(all_acr_values) if all_acr_values else None,
        'max': max(all_acr_values) if all_acr_values else None,
        'mean': np.mean(all_acr_values) if all_acr_values else None,
        'median': np.median(all_acr_values) if all_acr_values else None,
        'std': np.std(all_acr_values) if all_acr_values else None
    }

    return best_models_acr, best_models_aes, acr_stats, models


def analyze_single_distribution_models(models):
    """
    Analyze single distribution models (n_1) for each distribution type and spatial category.
    For Student's t models, find the one with the best degrees of freedom.
    """
    single_dist_models = {}
    best_tmm_models = {}

    for spatial_type, dist_types in models.items():
        single_dist_models[spatial_type] = {}
        best_tmm_models[spatial_type] = None

        # Process each distribution type
        for dist_type, model_list in dist_types.items():
            # Filter for n_1 models
            n1_models = [
                model for model in model_list if model['n_components'] == 1]

            if n1_models:
                if dist_type == 'tmm':
                    # For Student's t, find the best model based on ACR
                    sorted_models = sorted(
                        n1_models, key=lambda x: x['acr_distance'])
                    best_tmm_models[spatial_type] = sorted_models[0]
                else:
                    # For other distributions, just take the n_1 model
                    single_dist_models[spatial_type][dist_type] = n1_models[0]

    return single_dist_models, best_tmm_models


def compare_single_vs_mixture(single_dist_models, best_tmm_models, best_models):
    """
    Compare single distribution models with their mixture model counterparts.
    """
    comparison = {}

    for spatial_type in single_dist_models:
        comparison[spatial_type] = {}

        # Compare regular distributions
        for dist_type in single_dist_models[spatial_type]:
            single_model = single_dist_models[spatial_type][dist_type]
            best_mixture = best_models[spatial_type].get(dist_type)

            if best_mixture:
                single_metrics = single_model['metrics']
                mixture_metrics = best_mixture['metrics']

                # Calculate improvement percentages
                acr_improvement = (
                    (mixture_metrics['ACR'] - single_metrics['ACR']) / single_metrics['ACR']) * 100
                ail_improvement = (
                    (mixture_metrics['AIL'] - single_metrics['AIL']) / single_metrics['AIL']) * 100
                aes_improvement = (
                    (mixture_metrics['AES'] - single_metrics['AES']) / single_metrics['AES']) * 100

                comparison[spatial_type][dist_type] = {
                    'single_model': single_model,
                    'best_mixture': best_mixture,
                    'improvements': {
                        'acr': acr_improvement,
                        'ail': ail_improvement,
                        'aes': aes_improvement
                    }
                }

        # Compare TMM models
        if best_tmm_models[spatial_type] and 'tmm' in best_models[spatial_type]:
            single_model = best_tmm_models[spatial_type]
            best_mixture = best_models[spatial_type]['tmm']

            single_metrics = single_model['metrics']
            mixture_metrics = best_mixture['metrics']

            # Calculate improvement percentages
            acr_improvement = (
                (mixture_metrics['ACR'] - single_metrics['ACR']) / single_metrics['ACR']) * 100
            ail_improvement = (
                (mixture_metrics['AIL'] - single_metrics['AIL']) / single_metrics['AIL']) * 100
            aes_improvement = (
                (mixture_metrics['AES'] - single_metrics['AES']) / single_metrics['AES']) * 100

            comparison[spatial_type]['tmm'] = {
                'single_model': single_model,
                'best_mixture': best_mixture,
                'improvements': {
                    'acr': acr_improvement,
                    'ail': ail_improvement,
                    'aes': aes_improvement
                }
            }

    return comparison


def print_results(best_models_acr, best_models_aes, acr_stats):
    """
    Print the results in a formatted way.
    """
    print("\n" + "="*80)
    print("BEST MODELS (ACR closest to 0.8)")
    print("="*80)

    for spatial_type, dist_types in best_models_acr.items():
        print(f"\n{spatial_type.upper().replace('_', ' ')} MODELS:")
        print("-" * 50)

        for dist_type, model in dist_types.items():
            # Clean up distribution type name for display
            display_dist = dist_type.replace('_', ' ').upper()

            print(f"\n{display_dist}:")
            print(f"  Filename: {model['filename']}")
            print(f"  Components: {model['n_components']}")
            if model['nu'] is not None:
                print(f"  Nu: {model['nu']}")

            metrics = model['metrics']
            print(
                f"  ACR: {metrics['ACR']:.4f} (distance to 0.8: {model['acr_distance']:.4f})")
            print(f"  AIL: {metrics['AIL']:.4f}")
            print(f"  AES: {metrics['AES']:.4f}")

    # Print ACR statistics
    print("\n" + "="*80)
    print("ACR STATISTICS ACROSS ALL MODELS")
    print("="*80)
    print(f"Min ACR: {acr_stats['min']:.4f}")
    print(f"Max ACR: {acr_stats['max']:.4f}")
    print(f"Mean ACR: {acr_stats['mean']:.4f}")
    print(f"Median ACR: {acr_stats['median']:.4f}")
    print(f"Standard Deviation: {acr_stats['std']:.4f}")

    print("\n" + "="*80)
    print("BEST MODELS (Lowest AES)")
    print("="*80)

    for spatial_type, dist_types in best_models_aes.items():
        print(f"\n{spatial_type.upper().replace('_', ' ')} MODELS:")
        print("-" * 50)

        for dist_type, model in dist_types.items():
            # Clean up distribution type name for display
            display_dist = dist_type.replace('_', ' ').upper()

            print(f"\n{display_dist}:")
            print(f"  Filename: {model['filename']}")
            print(f"  Components: {model['n_components']}")
            if model['nu'] is not None:
                print(f"  Nu: {model['nu']}")

            metrics = model['metrics']
            print(
                f"  ACR: {metrics['ACR']:.4f} (distance to 0.8: {model['acr_distance']:.4f})")
            print(f"  AIL: {metrics['AIL']:.4f}")
            print(f"  AES: {metrics['AES']:.4f}")


def print_single_distribution_results(single_dist_models, best_tmm_models):
    """
    Print the results for single distribution models.
    """
    print("\n" + "="*80)
    print("SINGLE DISTRIBUTION MODELS (n=1)")
    print("="*80)

    for spatial_type in single_dist_models:
        print(f"\n{spatial_type.upper().replace('_', ' ')} MODELS:")
        print("-" * 50)

        # Print regular single distribution models
        for dist_type, model in single_dist_models[spatial_type].items():
            # Clean up distribution type name for display
            display_dist = dist_type.replace('_', ' ').upper()

            print(f"\n{display_dist}:")
            print(f"  Filename: {model['filename']}")
            metrics = model['metrics']
            print(f"  ACR: {metrics['ACR']:.4f}")
            print(f"  AIL: {metrics['AIL']:.4f}")
            print(f"  AES: {metrics['AES']:.4f}")

        # Print best TMM model
        if best_tmm_models[spatial_type]:
            model = best_tmm_models[spatial_type]
            print(f"\nBEST STUDENT'S T (TMM):")
            print(f"  Filename: {model['filename']}")
            print(f"  Nu (degrees of freedom): {model['nu']}")
            metrics = model['metrics']
            print(f"  ACR: {metrics['ACR']:.4f}")
            print(f"  AIL: {metrics['AIL']:.4f}")
            print(f"  AES: {metrics['AES']:.4f}")


def print_comparison_results(comparison):
    """
    Print the comparison between single distribution and mixture models.
    """
    print("\n" + "="*80)
    print("COMPARISON: SINGLE DISTRIBUTION vs MIXTURE MODELS")
    print("="*80)

    for spatial_type, dist_types in comparison.items():
        print(f"\n{spatial_type.upper().replace('_', ' ')} MODELS:")
        print("-" * 50)

        for dist_type, comp_data in dist_types.items():
            # Clean up distribution type name for display
            display_dist = dist_type.replace('_', ' ').upper()

            single_model = comp_data['single_model']
            best_mixture = comp_data['best_mixture']
            improvements = comp_data['improvements']

            print(f"\n{display_dist}:")
            print(f"  Single Distribution (n=1):")
            if dist_type == 'tmm':
                print(f"    Nu (degrees of freedom): {single_model['nu']}")
            print(f"    ACR: {single_model['metrics']['ACR']:.4f}")
            print(f"    AIL: {single_model['metrics']['AIL']:.4f}")
            print(f"    AES: {single_model['metrics']['AES']:.4f}")

            print(f"  Best Mixture (n={best_mixture['n_components']}):")
            if best_mixture['nu'] is not None:
                print(f"    Nu (degrees of freedom): {best_mixture['nu']}")
            print(f"    ACR: {best_mixture['metrics']['ACR']:.4f}")
            print(f"    AIL: {best_mixture['metrics']['AIL']:.4f}")
            print(f"    AES: {best_mixture['metrics']['AES']:.4f}")

            print(f"  Improvement (%):")
            print(f"    ACR: {improvements['acr']:.2f}%")
            print(f"    AIL: {improvements['ail']:.2f}%")
            print(f"    AES: {improvements['aes']:.2f}%")


def save_to_csv(best_models, filename="best_models.csv"):
    """
    Save the best models to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['spatial_type', 'distribution_type', 'n_components', 'nu',
                      'acr', 'acr_distance', 'ail', 'aes', 'filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for spatial_type, dist_types in best_models.items():
            for dist_type, model in dist_types.items():
                metrics = model['metrics']
                writer.writerow({
                    'spatial_type': spatial_type,
                    'distribution_type': dist_type,
                    'n_components': model['n_components'],
                    'nu': model['nu'] if model['nu'] is not None else 'N/A',
                    'acr': metrics['ACR'],
                    'acr_distance': model['acr_distance'],
                    'ail': metrics['AIL'],
                    'aes': metrics['AES'],
                    'filename': model['filename']
                })

    print(f"\nResults saved to {filename}")


def save_single_dist_to_csv(single_dist_models, best_tmm_models, filename="single_distribution_models.csv"):
    """
    Save the single distribution models to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['spatial_type', 'distribution_type',
                      'nu', 'acr', 'ail', 'aes', 'filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Write regular single distribution models
        for spatial_type, dist_types in single_dist_models.items():
            for dist_type, model in dist_types.items():
                metrics = model['metrics']
                writer.writerow({
                    'spatial_type': spatial_type,
                    'distribution_type': dist_type,
                    'nu': model['nu'] if model['nu'] is not None else 'N/A',
                    'acr': metrics['ACR'],
                    'ail': metrics['AIL'],
                    'aes': metrics['AES'],
                    'filename': model['filename']
                })

        # Write best TMM models
        for spatial_type, model in best_tmm_models.items():
            if model:
                metrics = model['metrics']
                writer.writerow({
                    'spatial_type': spatial_type,
                    'distribution_type': 'tmm (best nu)',
                    'nu': model['nu'],
                    'acr': metrics['ACR'],
                    'ail': metrics['AIL'],
                    'aes': metrics['AES'],
                    'filename': model['filename']
                })

    print(f"\nSingle distribution results saved to {filename}")


def save_comparison_to_csv(comparison, filename="comparison_single_vs_mixture.csv"):
    """
    Save the comparison between single distribution and mixture models to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'spatial_type', 'distribution_type',
            'single_n', 'single_nu', 'single_acr', 'single_ail', 'single_aes',
            'mixture_n', 'mixture_nu', 'mixture_acr', 'mixture_ail', 'mixture_aes',
            'acr_improvement_pct', 'ail_improvement_pct', 'aes_improvement_pct'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for spatial_type, dist_types in comparison.items():
            for dist_type, comp_data in dist_types.items():
                single_model = comp_data['single_model']
                best_mixture = comp_data['best_mixture']
                improvements = comp_data['improvements']

                writer.writerow({
                    'spatial_type': spatial_type,
                    'distribution_type': dist_type,
                    'single_n': single_model['n_components'],
                    'single_nu': single_model['nu'] if single_model['nu'] is not None else 'N/A',
                    'single_acr': single_model['metrics']['ACR'],
                    'single_ail': single_model['metrics']['AIL'],
                    'single_aes': single_model['metrics']['AES'],
                    'mixture_n': best_mixture['n_components'],
                    'mixture_nu': best_mixture['nu'] if best_mixture['nu'] is not None else 'N/A',
                    'mixture_acr': best_mixture['metrics']['ACR'],
                    'mixture_ail': best_mixture['metrics']['AIL'],
                    'mixture_aes': best_mixture['metrics']['AES'],
                    'acr_improvement_pct': improvements['acr'],
                    'ail_improvement_pct': improvements['ail'],
                    'aes_improvement_pct': improvements['aes']
                })

    print(f"\nComparison results saved to {filename}")


if __name__ == "__main__":
    best_models_acr, best_models_aes, acr_stats, all_models = analyze_results()
    print_results(best_models_acr, best_models_aes, acr_stats)
    save_to_csv(best_models_acr)

    # Analyze and print single distribution models
    single_dist_models, best_tmm_models = analyze_single_distribution_models(
        all_models)
    print_single_distribution_results(single_dist_models, best_tmm_models)
    save_single_dist_to_csv(single_dist_models, best_tmm_models)

    # Compare single distribution models with mixture models
    comparison = compare_single_vs_mixture(
        single_dist_models, best_tmm_models, best_models_acr)
    print_comparison_results(comparison)
    save_comparison_to_csv(comparison)
