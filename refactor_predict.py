import re
import sys

def main():
    path = "c:/Users/Administrator/baum/scripts/predict.py"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    skip = False
    
    for i, line in enumerate(lines):
        # Stop skipping after predict_order is done
        if skip and line.startswith("# ============================================================================"):
            if "# EXPLANATION" in lines[i+1]:
                skip = False
                
        if line.startswith("# ── Domain keyword patterns (mirrored from model_phase2.py) ──────────────────"):
            skip = True
            # Insert our imports before we start skipping
            new_lines.append("from repair_order.config import KEYWORD_FLAGS, TARGET_LABELS, TARGET_KEYWORD_MAP\n")
            new_lines.append("from repair_order.pipeline import load_pipeline as lib_load_pipeline, predict_order\n\n")
            new_lines.append("compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}\n\n")
            
        if not skip:
            # Replacements
            line = line.replace("result['total_input_time_min']", "result['total_input_time_hrs']")
            line = line.replace(" min\"", " hrs\"")
            line = line.replace(" min)", " hrs)")
            line = line.replace("Pred Total(min)", "Pred Total(hrs)")
            line = line.replace("True Total(min)", "True Total(hrs)")
            line = line.replace("Error(min)", "Error(hrs)")
            line = line.replace("total_predicted_minutes", "total_predicted_hours")
            
            # Refactoring load_pipeline
            if line.startswith("def load_pipeline(path):"):
                new_lines.append("def load_pipeline(path, data_path):\n")
                new_lines.append("    print(f\"  Loading pipeline from: {path}\")\n")
                new_lines.append("    pipeline = lib_load_pipeline(Path(path))\n")
                new_lines.append("    with open(data_path, \"r\", encoding=\"utf-8\") as f:\n")
                new_lines.append("        raw = json.load(f)\n")
                new_lines.append("    valid = [r for r in raw if \"input\" in r and \"output\" in r]\n")
                new_lines.append("    print(f\"  Pipeline loaded. Targets: {len(pipeline['output_targets'])}\")\n")
                new_lines.append("    return pipeline, valid\n")
                skip = True
                continue
                
            if skip and line.startswith("# ============================================================================"):
                if "# MAIN" in lines[i+1]:
                    skip = False
                    
            if not skip:
                new_lines.append(line)
                
    # Fix argparse in main
    main_code = "".join(new_lines)
    
    main_code = main_code.replace(
        "    parser = argparse.ArgumentParser(\n        description=\"Predict repair work steps from a JSON order.\"\n    )",
        "    parser = argparse.ArgumentParser(description=\"Predict repair work steps from a JSON order.\")\n" + \
        "    parser.add_argument(\"--data\", type=str, default=str(ROOT / \"data\" / \"orders_simplified_sample.json\"))\n" + \
        "    parser.add_argument(\"--model\", type=str, default=str(ROOT / \"models\" / \"two_stage_pipeline.pkl\"))"
    )
    
    main_code = main_code.replace(
        "    pipeline, dataset = load_pipeline(PIPELINE_PATH)",
        "    pipeline, dataset = load_pipeline(args.model, args.data)"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(main_code)

if __name__ == "__main__":
    main()
