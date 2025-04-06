import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import subprocess
import whisperx

def convert_to_ct2(model_path, output_dir):
    print(f"Converting model '{model_path}' to CTranslate2 format...")
    command = [
        "ct2-transformers-converter",
        "--model", model_path,
        "--output_dir", output_dir,
        "--copy_files", "tokenizer_config.json", "preprocessor_config.json", "special_tokens_map.json", "added_tokens.json",
        "--quantization", "float16"
    ]
    subprocess.run(command, check=True)
    print("Conversion complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Convert and transcribe with Whisper via CTranslate2 backend.")
    parser.add_argument("--model_path", type=str, default="openai/whisper-large-v3-turbo", help="Path to Whisper model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (e.g., mp3 or wav)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to run model on")
    parser.add_argument("--compute_type", type=str, choices=["float16", "int8", "int8_float16"], default="float16", help="Precision mode for inference")
    args = parser.parse_args()

    model_path = args.model_path
    model_name = os.path.basename(model_path)
    output_dir = f"ctranslate2-models/{model_name}"

    if not os.path.isdir(output_dir):
        convert_to_ct2(model_path, output_dir)

    print(f"Loading model from {output_dir} on {args.device} ({args.compute_type})...")
    model = whisperx.load_model(output_dir, device=args.device, compute_type=args.compute_type)

    print(f"\nTranscribing: {args.audio}")
    batch_size = 8
    segments, info = model.transcribe(args.audio, batch_size=batch_size)

    print(f"\nDetected language: '{info.language}' (probability {info.language_probability:.2f})\n")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    main()
