import os
import json


def get_best_acr_json(results_dir, target=0.8):
    best_file_acr = None
    best_diff = float('inf')
    best_acr = 0

    best_file_aes = None
    best_aes = float('inf')

    results = []

    tmm_total_aes = 0.0
    tmm_count = 0
    gmm_total_aes = 0.0
    gmm_count = 0

    # Loop through every file in the directory.
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)

                    acr = data.get("ACR")
                    aes = data.get("AES")

                    # Update best file for ACR based on target.
                    if acr is not None:
                        diff = abs(acr - target)
                        if diff < best_diff:
                            best_diff = diff
                            best_file_acr = filepath
                            best_acr = acr

                    # Update best file for AES (lower is better).
                    if aes is not None:
                        if aes < best_aes:
                            best_aes = aes
                            best_file_aes = filepath

                    # Compute total AES and counts based on filename keywords.
                    lower_fname = filename.lower()
                    if "tmm" in lower_fname:
                        if aes is not None:
                            tmm_total_aes += aes
                            tmm_count += 1
                    elif "gmm" in lower_fname:
                        if aes is not None:
                            gmm_total_aes += aes
                            gmm_count += 1

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    avg_aes_tmm = tmm_total_aes / tmm_count if tmm_count > 0 else None
    avg_aes_gmm = gmm_total_aes / gmm_count if gmm_count > 0 else None

    return best_acr, best_file_acr, best_aes, best_file_aes, results, avg_aes_tmm, avg_aes_gmm


# Example usage:
if __name__ == '__main__':
    results_folder = 'results'
    best_acr, best_acr_file, best_aes, best_aes_file, all_results, avg_aes_tmm, avg_aes_gmm = get_best_acr_json(
        results_folder)

    if best_acr_file:
        print(
            f"The JSON file with the best ACR (closest to 0.8) is: {best_acr_file} with {best_acr}")
    else:
        print("No valid JSON file with an ACR value was found.")

    if best_aes_file:
        print(
            f"The JSON file with the best (lowest) AES is: {best_aes_file} with {best_aes}")
    else:
        print("No valid JSON file with an AES value was found.")

    print("Average AES for TMM files:", avg_aes_tmm)
    print("Average AES for GMM files:", avg_aes_gmm)

    print("\nAll result dictionaries:")
    for res in all_results:
        print(res)
