import os
import argparse
import transformers
import ctranslate2
import subprocess

LANGUAGES = {
   'hi': "hin_Deva", 
   'id': "ind_Latn", 
   'ms': "zsm_Latn",
   'th': "tha_Thai",
   'tl': "tgl_Latn",
   'vi': "vie_Latn",
   'zh-CN': "zho_Hans",
   'en': "eng_Latn"
}

def main():
    parser = argparse.ArgumentParser(description="Convert and use a CTranslate2 model for translation.")
    parser.add_argument("--model_path", type=str, help="Path to the model directory. If omitted, uses the default Facebook model.")
    parser.add_argument("--src_lang", type=str, choices=LANGUAGES.keys(), help="Source language code.")
    parser.add_argument("--tgt_lang", type=str, choices=LANGUAGES.keys(), help="Target language code.")
    parser.add_argument("--input", type=str, help="Input string to translate.")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path else "facebook/nllb-200-distilled-600M"
    model_name = os.path.basename(model_path)  # Take as output directory
    
    output_path = "ctranslate2-models/" + model_name

    if not os.path.isdir(output_path):
        command = ["ct2-transformers-converter", "--model", model_path, "--output_dir", output_path]
        subprocess.call(command)

    input_str = args.input
    src_lang = LANGUAGES[args.src_lang]
    tgt_lang = LANGUAGES[args.tgt_lang]

    translator = ctranslate2.Translator(output_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_str))
    target_prefix = [tgt_lang]
    results = translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]

    print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))


if __name__ == "__main__":
    main()
