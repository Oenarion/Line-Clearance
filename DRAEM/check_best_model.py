import os

# Path to your config file
results_path = "DRAEM\\results"

# List of categories
categories = ["A", "A2", "B", "B2", "C"]
load_chps = [2000,2250,2500,2750,3000]
masked = ['masking', 'no_masking']
thresholds_type = ['k-sigma_threshold', 'max_threshold', 'p-quantile_threshold']
img_reg = ['image_reg', 'no_image_reg']
masking = ['pre_masking', 'post_masking']


def compute_best_metric(metric_map, idx):
    """
    Computes the best score and the best checkpoint(s).

    Args:
        - metric_map: map with the metrics saved
        - idx: int -> 0 img_auroc, 1 pixel_auroc, 2 pro, 3 iou, 4 precision recall, 5 accuracy
    """
    best_score = -9999
    best_checkpoints = []
    best_thresholds = []

    for chp in load_chps:
        for thresh in thresholds_type:
            score = metric_map[chp][thresh][idx]
            if score > best_score:
                best_score = score

    if idx in [0, 1, 2, 4]:
        for chp in load_chps:
            # THESE METRICS ARE EQUAL FOR ALL THRESHOLD, NO POINT IN EVALUATING ALL THE DIFFERENT THRESHOLDS!
            if metric_map[chp][thresholds_type[0]][idx] == best_score:
                best_checkpoints.append(chp)
    else:
        for chp in load_chps:
            for thresh in thresholds_type:
                if metric_map[chp][thresh][idx] == best_score:
                    best_checkpoints.append(chp)
                    best_thresholds.append(thresh)

    return best_score, best_checkpoints, best_thresholds

def save_to_file(best_metrics_mask, best_metrics_no_mask, save_path):
    """
    Prints best metrics for masked and non masked images, for different thresholds and checkpoints in a file
    """

    with open(save_path, 'w') as f:
    
        f.write("RESULTS WITH MASKING \n")
        f.write(f"BEST IMG AUROC: {best_metrics_mask['img_auroc'][0]}, BEST CHECKPOINTS: {best_metrics_mask['img_auroc'][1]} \n")
        f.write(f"BEST PIXEL AUROC: {best_metrics_mask['pixel_auroc'][0]}, BEST CHECKPOINTS: {best_metrics_mask['pixel_auroc'][1]} \n")
        f.write(f"BEST PRO: {best_metrics_mask['pro'][0]}, BEST CHECKPOINTS: {best_metrics_mask['pro'][1]} \n")
        f.write(f"BEST IOU: {best_metrics_mask['iou'][0]}, BEST CHECKPOINTS: {best_metrics_mask['iou'][1]}, BEST THRESHOLDS_TYPE: {best_metrics_mask['iou'][2]} \n")
        f.write(f"BEST PRECISION RECALL: {best_metrics_mask['pr'][0]}, BEST CHECKPOINTS: {best_metrics_mask['pr'][1]} \n")
        f.write(f"BEST ACCURACY: {best_metrics_mask['acc'][0]}, BEST CHECKPOINTS: {best_metrics_mask['acc'][1]}, BEST THRESHOLDS_TYPE: {best_metrics_mask['acc'][2]} \n")

        f.write("="*100)
        f.write('\n')

        f.write("RESULTS WITH NO MASKING \n")    
        f.write(f"BEST IMG AUROC: {best_metrics_no_mask['img_auroc'][0]}, BEST CHECKPOINTS: {best_metrics_no_mask['img_auroc'][1]} \n")
        f.write(f"BEST PIXEL AUROC: {best_metrics_no_mask['pixel_auroc'][0]}, BEST CHECKPOINTS: {best_metrics_no_mask['pixel_auroc'][1]} \n")
        f.write(f"BEST PRO: {best_metrics_no_mask['pro'][0]}, BEST CHECKPOINTS: {best_metrics_no_mask['pro'][1]} \n")
        f.write(f"BEST IOU: {best_metrics_no_mask['iou'][0]}, BEST CHECKPOINTS: {best_metrics_no_mask['iou'][1]}, BEST THRESHOLDS_TYPE: {best_metrics_no_mask['iou'][2]} \n")
        f.write(f"BEST PRECISION RECALL: {best_metrics_no_mask['pr'][0]}, BEST CHECKPOINTS: {best_metrics_no_mask['pr'][1]} \n")
        f.write(f"BEST ACCURACY: {best_metrics_no_mask['acc'][0]}, BEST CHECKPOINTS: {best_metrics_no_mask['acc'][1]}, BEST THRESHOLDS_TYPE: {best_metrics_no_mask['acc'][2]} \n")

def best_metric_call(metric_map):
    """
    Calls the function to compute the best metric for each metric.

    Args:
        - metric_map: map with the metrics saved
    """
    best_metrics = {}
    img_auroc_best, img_auroc_best_chp, img_auroc_best_thresh = compute_best_metric(metric_map, 0)
    pixel_auroc_best, pixel_auroc_best_chp, pixel_auroc_best_thresh = compute_best_metric(metric_map, 1)
    pro_best, pro_best_chp, pro_best_thresh = compute_best_metric(metric_map, 2)
    iou_best, iou_best_chp, iou_best_thresh = compute_best_metric(metric_map, 3)
    pr_best, pr_best_chp, pr_best_thresh = compute_best_metric(metric_map, 4)
    acc_best, acc_best_chp, acc_best_thresh = compute_best_metric(metric_map, 5)
    best_metrics['img_auroc'] = [img_auroc_best, img_auroc_best_chp, img_auroc_best_thresh]
    best_metrics['pixel_auroc'] = [pixel_auroc_best, pixel_auroc_best_chp, pixel_auroc_best_thresh]
    best_metrics['pro'] = [pro_best, pro_best_chp, pro_best_thresh]
    best_metrics['iou'] = [iou_best, iou_best_chp, iou_best_thresh]
    best_metrics['pr'] = [pr_best, pr_best_chp, pr_best_thresh]
    best_metrics['acc'] = [acc_best, acc_best_chp, acc_best_thresh]

    return best_metrics

def main():
    """Main script to automatically get the best scores out of all the trained networks."""
    print("-"*50)
    for mask_type in masking:
        for reg in img_reg:
            for category in categories:
                mask_results = {}
                non_mask_results = {}
                for chp in load_chps:
                    mask_results[chp] = {}
                    non_mask_results[chp] = {}
                    for thresh in thresholds_type:
                        for is_mask in masked:
                            path = os.path.join(os.getcwd(),results_path, mask_type, reg, category, f'checkpoint_{chp}', thresh,is_mask, 'metrics.txt')
                            with open(path, "r") as f:
                                lines = f.readlines()
                                auroc = lines[1].split(':')[1]
                                img_auroc = float(auroc.split('(')[1].split(',')[0])
                                pixel_auroc = float(auroc.split(',')[1].split(')')[0])
                                pro = float(lines[2].split(':')[1].split('\n')[0])                    
                                iou = float(lines[3].split(':')[1].split('\n')[0])
                                ap = float(lines[4].split(':')[1].split('\n')[0])
                                acc = float(lines[5].split(':')[1].split('%')[0])
                                if is_mask == 'masking':
                                    mask_results[chp][thresh] = [img_auroc, pixel_auroc, pro, iou, ap, acc]
                                else:
                                    non_mask_results[chp][thresh] = [img_auroc, pixel_auroc, pro, iou, ap, acc]
                        
                        # print(mask_results)
                        # print("*"*100)
                best_metrics_mask = best_metric_call(mask_results)
                best_metrics_no_mask = best_metric_call(non_mask_results)
                save_path = os.path.join(os.getcwd(),results_path, mask_type, reg, category, 'metrics_report.txt')
                
                save_to_file(best_metrics_mask, best_metrics_no_mask, save_path)

if __name__ == "__main__":
    main()
